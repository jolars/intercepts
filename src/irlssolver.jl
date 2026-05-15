using LinearAlgebra
using Statistics
using Random
using SparseArrays

abstract type IRLSWeights end
struct LocalWeights <: IRLSWeights end
struct MajorizedWeights <: IRLSWeights end

# Per-observation weights for the IRLS quadratic majorizer at iterate η.
function _irls_weights(
    ::LocalWeights,
    lossfun::LogisticLoss,
    η::AbstractVector,
    y::AbstractVector,
)
    return hessian(lossfun, η, y)
end

function _irls_weights(
    ::MajorizedWeights,
    lossfun::LogisticLoss,
    η::AbstractVector,
    ::AbstractVector,
)
    return fill(lossfun.lipschitz, length(η))
end

# Inner-quadratic intercept update.
# All non-NoIntercept strategies coincide on the quadratic majorizer: Newton is
# exact, the gradient step with the inner Lipschitz (= curvature = sum(w))
# matches it, ExactStrategy reaches the same point in one step, and a
# damped Newton step with backtracking accepts the full step. The dispatch is
# preserved so NoIntercept still disables the update.
_irls_inner_intercept_step(::NoIntercept, grad_0::Real, curv::Real) = 0.0
function _irls_inner_intercept_step(::InterceptStrategy, grad_0::Real, curv::Real)
    return curv > 0 ? -grad_0 / curv : 0.0
end

function irlssolver(
    x::AbstractMatrix,
    y::AbstractVector,
    reg::Real = 0.1;
    lossfun::LogisticLoss = LogisticLoss(),
    weights::IRLSWeights = LocalWeights(),
    intercept_strategy::InterceptStrategy = NewtonStrategy(),
    outer_tol::Real = 1.0e-10,
    inner_tol::Real = 1.0e-4,
    max_outer::Int = 200,
    max_inner::Int = 50,
    maxtime::Real = Inf,
    randomize::Bool = true,
    normalization::Symbol = :standardize,
    save_history::Bool = false,
)
    n, p = size(x)
    y = Float64.(y)

    validateresponse(lossfun, y)

    x, x_centers, x_scales = normalizefeatures(x, normalization)
    sparse_norm = issparse(x) && normalization != :none
    x_sparse_offset = x_centers ./ x_scales

    fit_intercept = !(intercept_strategy isa NoIntercept)

    λmax = lambdamax(lossfun, x, y)
    λ = reg * λmax

    intercept = 0.0
    coef = zeros(p)
    η = zeros(n)

    intercepts = Float64[]
    coefs = Vector{Vector{Float64}}(undef, 0)
    primals = Float64[]
    duals = Float64[]
    gaps = Float64[]
    relgaps = Float64[]
    times = Float64[]

    t0 = time()
    it = 0

    while it < max_outer && (time() - t0) < maxtime
        it += 1

        # Outer-level primal/dual gap on the original loss (matches cdsolver).
        primal = loss(lossfun, η, y) + λ * norm(coef, 1)
        r = residual(lossfun, η, y)
        θ = copy(r)
        if fit_intercept
            θ .-= mean(r)
        end
        grad_outer = x' * θ
        if sparse_norm
            θsum = sum(θ)
            for j = 1:p
                grad_outer[j] -= x_sparse_offset[j] * θsum
            end
        end
        dual_scale = max(1, norm(grad_outer, Inf) / λ)
        θ ./= dual_scale
        dua = dual(lossfun, θ, y)
        gap = primal - dua
        relgap = gap / max(abs(primal), 1.0e-10)

        push!(primals, primal)
        push!(duals, dua)
        push!(gaps, gap)
        push!(relgaps, relgap)
        push!(times, time() - t0)

        if save_history
            int_rs, coef_rs = rescalecoefs(
                coef,
                intercept,
                x_centers,
                x_scales;
                fit_intercept = fit_intercept,
            )
            push!(intercepts, int_rs)
            push!(coefs, coef_rs)
        end

        if relgap <= outer_tol
            break
        end

        # Linearize: form the weighted-LS majorizer at the current iterate.
        # w_i: per-observation curvature; z_i = η_i - r_i / w_i is the working response,
        # written here as η + (y - μ)/w to keep numerator and denominator co-signed.
        w = _irls_weights(weights, lossfun, η, y)
        w_safe = max.(w, 1.0e-12)
        μ = invlink(lossfun, η)
        z = η .+ (y .- μ) ./ w_safe

        # H_00 of the inner quadratic, constant within the inner solve.
        inner_curv = sum(w)
        if inner_curv <= 0
            inner_curv = 1.0e-12
        end

        η_in = copy(η)
        coef_in = copy(coef)
        intercept_in = intercept

        prev_inner_obj = Inf

        for _ = 1:max_inner
            ind = randomize ? randperm(p) : (1:p)

            for j in ind
                xj = x[:, j]
                resid_in = η_in .- z
                grad_j = sum(w .* xj .* resid_in)
                hess_j = sum(w .* xj .^ 2)
                if sparse_norm
                    grad_j -= x_sparse_offset[j] * sum(w .* resid_in)
                    hess_j -=
                        2 * x_sparse_offset[j] * sum(w .* xj) -
                        x_sparse_offset[j]^2 * sum(w)
                end
                if hess_j <= 0
                    hess_j = 1.0e-10
                end
                old_coef = coef_in[j]
                coef_in[j] = st(old_coef - grad_j / hess_j, λ / hess_j)
                Δ = coef_in[j] - old_coef
                if Δ != 0
                    η_in .+= Δ * xj
                    if sparse_norm
                        η_in .-= Δ * x_sparse_offset[j]
                    end
                end
            end

            if fit_intercept
                grad_0 = sum(w .* (η_in .- z))
                Δ_int = _irls_inner_intercept_step(intercept_strategy, grad_0, inner_curv)
                if Δ_int != 0
                    intercept_in += Δ_int
                    η_in .+= Δ_int
                end
            end

            inner_obj = 0.5 * sum(w .* (η_in .- z) .^ 2) + λ * norm(coef_in, 1)
            if abs(prev_inner_obj - inner_obj) <= inner_tol * max(1.0, abs(inner_obj))
                break
            end
            prev_inner_obj = inner_obj
        end

        coef = coef_in
        intercept = intercept_in
        η = η_in
    end

    intercept_rs, coef_rs = rescalecoefs(
        coef,
        intercept,
        x_centers,
        x_scales;
        fit_intercept = fit_intercept,
    )

    return (
        intercept = intercept_rs,
        coef = coef_rs,
        primals = primals,
        duals = duals,
        gaps = gaps,
        relgaps = relgaps,
        time = times,
        passes = it,
        coefs = save_history ? reduce(hcat, coefs) : Matrix{Float64}(undef, 0, 0),
        intercepts = intercepts,
        λ = λ,
        λmax = λmax,
    )
end
