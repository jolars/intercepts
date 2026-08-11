using Intercepts
using Random
using ProjectRoot
using JLD2
using LinearAlgebra
using Statistics

# Isolates the ρ_{0j}^2 axis of the Schur framing. At fixed μ_0 (which pins
# H_00/L_0), toggling feature centering moves the weighted-average ρ̄² between
# small (standardized) and large (uncentered, with O(1) per-column means from
# generatedata's `means = :random`). The eq-rate-gap leading-order prediction
# is then (1 - (H_00/L_0) ρ̄²) / (1 - ρ̄²); we compare it to empirical pass
# counts T_G / T_N for the gradient and Newton strategies at a fixed tolerance.

const N = 500
const P = 1000
const S = 10
const REG = 0.05
const Μ0_GRID = [0.5, 0.9]
const NORM_GRID = [:standardize, :none]
const TOL_TARGET = 1.0e-6
const MAXIT = 30_000

function standardize_features(X)
    centers = mean(X; dims = 1)
    scales = stdm(X, centers; corrected = false, dims = 1)
    scales[scales .== 0] .= 1.0
    return (X .- centers) ./ scales
end

function rate_quantities(X_used, η, lossfun)
    w = weight(lossfun, η)
    H00 = sum(w)
    Hjj = vec(sum((X_used.^2) .* w; dims = 1))
    H0j = vec(sum(X_used .* w; dims = 1))
    ρ2 = (H0j.^2) ./ (H00 .* Hjj)
    barρ2 = sum(Hjj .* ρ2) / sum(Hjj)
    return (; H00, Hjj, H0j, ρ2, barρ2, w)
end

passes_to(relgaps, tol) = let i = findfirst(g -> g ≤ tol, relgaps)
    i === nothing ? length(relgaps) : i
end

records = Dict{String, Any}[]

for μ0 in Μ0_GRID, normalization in NORM_GRID
    Random.seed!(1)
    X_raw, y = generatedata(
        N,
        P;
        response = :binomial,
        μ0 = μ0,
        x_type = :normal,
        x_density = 0.9,
        ρ = 0.6,
        s = S,
        amplitude = 1.0,
        means = :random,
    )

    X_used = normalization == :standardize ? standardize_features(X_raw) : X_raw

    lossfun = LogisticLoss()
    L0 = lossfun.lipschitz * N # global Lipschitz on intercept: n * sup f''

    # Reference Newton run for η at convergence (used to evaluate H_jj, ρ²
    # at the iterate-level Hessian, matching how sim-rate-gap.jl reports them).
    Random.seed!(1)
    newton_ref = cdsolver(
        X_raw,
        y,
        REG;
        lossfun = lossfun,
        intercept_strategy = NewtonStrategy(),
        maxit = MAXIT,
        randomize = true,
        tol = 1.0e-12,
        normalization = normalization,
        save_history = false,
    )

    # η is invariant to feature scaling: cdsolver rescales coefs back to the
    # original X scale before returning, so η = X β + β_0 is the natural one.
    η_hat = X_raw * newton_ref.coef .+ newton_ref.intercept
    q = rate_quantities(X_used, η_hat, lossfun)

    predicted_ratio = (1 - (q.H00 / L0) * q.barρ2) / (1 - q.barρ2)

    Random.seed!(1)
    grad_res = cdsolver(
        X_raw,
        y,
        REG;
        lossfun = lossfun,
        intercept_strategy = GradientStrategy(),
        maxit = MAXIT,
        randomize = true,
        tol = TOL_TARGET,
        normalization = normalization,
    )

    Random.seed!(1)
    newton_short = cdsolver(
        X_raw,
        y,
        REG;
        lossfun = lossfun,
        intercept_strategy = NewtonStrategy(),
        maxit = MAXIT,
        randomize = true,
        tol = TOL_TARGET,
        normalization = normalization,
    )

    T_grad = passes_to(grad_res.relgaps, TOL_TARGET)
    T_newt = passes_to(newton_short.relgaps, TOL_TARGET)
    grad_reached = any(g -> g ≤ TOL_TARGET, grad_res.relgaps)
    newt_reached = any(g -> g ≤ TOL_TARGET, newton_short.relgaps)

    empirical_ratio = T_grad / T_newt

    push!(
        records,
        Dict{String, Any}(
            "μ0" => μ0,
            "normalization" => String(normalization),
            "n" => N,
            "p" => P,
            "s" => S,
            "reg" => REG,
            "tol" => TOL_TARGET,
            "L0" => L0,
            "H00" => q.H00,
            "H00_over_L0" => q.H00 / L0,
            "barρ2" => q.barρ2,
            "predicted_ratio" => predicted_ratio,
            "T_newton" => T_newt,
            "T_gradient" => T_grad,
            "empirical_ratio" => empirical_ratio,
            "newton_reached_tol" => newt_reached,
            "gradient_reached_tol" => grad_reached,
        ),
    )

    @info "μ0=$(μ0) norm=$(normalization)  H00/L0=$(round(q.H00 / L0; sigdigits = 3))  ρ̄²=$(round(q.barρ2; sigdigits = 3))  predicted=$(round(predicted_ratio; sigdigits = 3))  empirical=$(round(empirical_ratio; sigdigits = 3))  (T_N=$(T_newt), T_G=$(T_grad))"
end

outfile = @projectroot("results", "rho-centering.jld2")
@save outfile records
println("Saved results to $outfile")
