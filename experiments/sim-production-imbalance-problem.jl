using Intercepts
using Random
using LinearAlgebra
using Statistics
using ProjectRoot
using CSV
using DataFrames
using JSON3

# Calibrate marginal prevalence rather than the probability at a zero feature
# vector, so the horizontal axis has the same meaning at every signal level.
function prevalence_intercept(signal, prevalence)
    lower, upper = -50.0 - maximum(signal), 50.0 - minimum(signal)
    for _ in 1:100
        midpoint = (lower + upper) / 2
        if mean(Intercepts.sigmoid.(signal .+ midpoint)) < prevalence
            lower = midpoint
        else
            upper = midpoint
        end
    end
    return (lower + upper) / 2
end

# Once CD identifies the active coefficients, a joint Newton refinement avoids
# losing reference accuracy to cancellation in individual coordinate updates.
function refine_reference(X, y, fit)
    active = findall(!iszero, fit.coef)
    Z = hcat(ones(length(y)), X[:, active])
    coefficients = vcat(fit.intercept, fit.coef[active])
    signs = sign.(coefficients[2:end])
    for _ in 1:5
        probabilities = Intercepts.sigmoid.(Z * coefficients)
        gradient = Z' * (probabilities .- y) + vcat(0.0, fit.λ .* signs)
        weights = probabilities .* (1 .- probabilities)
        step = Symmetric(Z' * (weights .* Z)) \ gradient
        candidate = coefficients - step
        all(sign.(candidate[2:end]) .== signs) ||
            error("Reference refinement changed the active orthant")
        coefficients = candidate
        norm(gradient, Inf) <= 1.0e-13 * max(1, fit.λ) && break
    end
    beta = zeros(size(X, 2))
    beta[active] = coefficients[2:end]
    return merge(fit, (; coef = beta, intercept = coefficients[1]))
end

# Saturated probabilities can make the Float64 dual-domain safeguard too
# conservative for a tight reference. Certify the fitted coefficients with
# higher precision, including the unpenalized-intercept equality constraint.
function certify_reference(X, y, fit)
    return setprecision(128) do
        XB, yB = BigFloat.(X), BigFloat.(y)
        beta = BigFloat.(fit.coef)
        intercept = BigFloat(fit.intercept)
        signal = XB * beta
        for _ in 1:20
            probabilities = Intercepts.sigmoid.(signal .+ intercept)
            step = sum(probabilities .- yB) / sum(probabilities .* (1 .- probabilities))
            intercept -= step
            abs(step) < big"1e-30" && break
        end
        eta = signal .+ intercept
        residual = Intercepts.sigmoid.(eta) .- yB
        residual .-= mean(residual)
        response_mean = mean(yB)
        contraction = one(BigFloat)
        for i in eachindex(yB)
            direction = yB[i] + residual[i] - response_mean
            if direction < 0
                contraction = min(contraction, response_mean / -direction)
            elseif direction > 0
                contraction = min(contraction, (1 - response_mean) / direction)
            end
        end
        contraction *= 1 - 64eps(BigFloat)
        anchor = response_mean .- yB
        theta = anchor .+ contraction .* (residual .- anchor)
        lambda = BigFloat(fit.λ)
        theta ./= max(one(BigFloat), norm(XB' * theta, Inf) / lambda) *
            (1 + 64eps(BigFloat))
        probabilities = yB .+ theta
        all(0 .< probabilities .< 1) || error("Dual point outside logistic domain")
        abs(sum(theta)) < big"1e-30" || error("Dual intercept constraint failed")
        norm(XB' * theta, Inf) <= lambda * (1 + big"1e-30") ||
            error("Dual coefficient constraint failed")
        n = length(y)
        primal = (sum(Intercepts.log1pexp.(eta) .- yB .* eta) + lambda * norm(beta, 1)) / n
        dual = -sum(
            probabilities .* log.(probabilities) .+
                (1 .- probabilities) .* log1p.(-probabilities),
        ) /
            n
        # Round the stored lower bound downward so conversion cannot overstate it.
        dual_bound = Float64(dual) - 64eps(Float64)
        return (; primal = Float64(primal), dual_bound, intercept = Float64(intercept))
    end
end

function reference_and_curvature(X, y, reg)
    fit = cdsolver(
        X,
        y,
        reg;
        lossfun = LogisticLoss(),
        intercept_strategy = NewtonStrategy(),
        normalization = :none,
        randomize = false,
        coef_init = zeros(size(X, 2)),
        intercept_init = log(mean(y) / (1 - mean(y))),
        tol = 1.0e-6,
        maxit = 1000,
    )
    fit = refine_reference(X, y, fit)
    n = length(y)
    certified = certify_reference(X, y, fit)
    primal, dual_bound = certified.primal, certified.dual_bound
    certificate = (primal - dual_bound) / primal
    0 <= certificate <= 1.0e-8 || error("Reference certificate failed: $certificate")

    eta = X * fit.coef .+ certified.intercept
    probabilities = Intercepts.sigmoid.(eta)
    weights = probabilities .* (1 .- probabilities)
    H00 = mean(weights)
    H0 = X' * weights / n
    Hdiag = vec(sum((X.^2) .* weights; dims = 1)) / n
    rho_squared = H0.^2 ./ (H00 .* Hdiag)
    active = findall(abs.(fit.coef) .> 1.0e-8)
    kappa = if isempty(active)
        0.0
    else
        XA = X[:, active]
        HAA = Symmetric(XA' * (weights .* XA) / n)
        dot(H0[active], HAA \ H0[active]) / H00
    end
    -1.0e-10 <= kappa <= 1 + 1.0e-10 || error("Invalid coupling: $kappa")
    return (;
        lambda = fit.λ / n,
        reference_primal = primal,
        dual_bound,
        reference_relative_gap = certificate,
        H00,
        curvature_fraction = 4H00,
        max_rho_squared = maximum(rho_squared),
        kappa,
        n_active = length(active),
        reference_passes = fit.passes,
    )
end

function main()
    n, p, s = 2000, 500, 10
    correlation, amplitude, reg = 0.6, 0.5, 0.05
    prevalences = [0.5, 0.7, 0.9, 0.95, 0.99]
    seeds = [1, 2, 3]
    outdir = @projectroot("results", "production-imbalance")
    mkpath(joinpath(outdir, "inputs"))
    records = NamedTuple[]
    BLAS.set_num_threads(1)

    for seed in seeds
        rng = MersenneTwister(seed)
        X = randn(rng, n, p)
        for j in 2:p
            X[:, j] .= correlation .* X[:, j - 1] .+ sqrt(1 - correlation^2) .* X[:, j]
        end
        X .-= mean(X; dims = 1)
        X ./= std(X; dims = 1, corrected = false)
        beta = zeros(p)
        beta[round.(Int, range(1, p; length = s))] .= amplitude
        signal = X * beta
        # Common uniforms make the responses nested as prevalence increases.
        uniforms = rand(rng, n)
        design_file = "seed_$(seed)_X.csv"
        CSV.write(
            joinpath(outdir, "inputs", design_file),
            DataFrame(X, :auto);
            writeheader = false,
        )

        for (level, prevalence) in enumerate(prevalences)
            intercept = prevalence_intercept(signal, prevalence)
            probabilities = Intercepts.sigmoid.(signal .+ intercept)
            y = Float64.(uniforms .< probabilities)
            0 < sum(y) < n || error("Both response classes are required")
            cell = "seed_$(seed)_level_$(level)"
            response_file = "$(cell)_y.csv"
            CSV.write(joinpath(outdir, "inputs", response_file), DataFrame(y = y))
            reference = reference_and_curvature(X, y, reg)
            lambda_max = norm(X' * (y .- mean(y)), Inf) / n
            isapprox(reference.lambda, reg * lambda_max; rtol = 1.0e-12) ||
                error("Lambda scaling mismatch")
            push!(
                records,
                (;
                cell, seed, prevalence, observed_prevalence = mean(y),
                n, p, s, correlation, amplitude, reg, generating_intercept = intercept,
                n_positive = Int(sum(y)), n_negative = Int(n - sum(y)),
                design_file, response_file, lambda_max, reference...,
            ),
            )
            CSV.write(joinpath(outdir, "problems.csv"), DataFrame(records))
            println(
                "$cell prevalence=$prevalence observed=$(mean(y)) q=$(reference.curvature_fraction) kappa=$(reference.kappa)",
            )
            flush(stdout)
        end
    end
    open(joinpath(outdir, "design.json"), "w") do io
        JSON3.pretty(
            io,
            (;
            n, p, s, correlation, amplitude, reg, prevalences, seeds,
            response = "Bernoulli with calibrated marginal prevalence and common uniforms",
            lambda_convention = "mean logistic loss + lambda * norm(beta, 1)",
            reference = "Guarded-Newton CD, active-set Newton refinement, 128-bit primal-dual certification",
            reference_relative_gap_limit = 1.0e-8,
            julia_version = string(VERSION),
        ),
        )
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
