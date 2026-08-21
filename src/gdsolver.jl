using LinearAlgebra
using Statistics
using Random
using Arpack

"""
    gdsolver(x, y, reg = 0.1; kwargs...)

Fit an L1-regularized generalized linear model by proximal gradient descent.
This reference solver uses a global Lipschitz step size and updates the
intercept after each gradient step. The result contains fitted values and the
per-pass `primals`, `duals`, `gaps`, `time`, and `passes` diagnostics.
"""
function gdsolver(
        x::AbstractMatrix,
        y::AbstractVector,
        reg::Real = 0.1;
        lossfun::LossFunction = QuadraticLoss(),
        intercept_strategy::InterceptStrategy = GradientStrategy(),
        tol::Real = 1.0e-10,
        normalization::Symbol = :standardize,
        maxit::Int = 1000,
    )
    n, p = size(x)

    validateresponse(lossfun, y)

    x, x_centers, x_scales = normalizefeatures(x, normalization)

    if issparse(x) && normalization != :none
        throw(ArgumentError("Sparse matrices with normalization are not supported."))
    end

    fit_intercept = !(intercept_strategy isa NoIntercept)

    λmax = lambdamax(lossfun, x, y)
    λ = reg * λmax

    intercept = 0.0
    coef = zeros(p)
    η = zeros(n)
    r = residual(lossfun, η, y)

    if issparse(x)
        L = svds(x, nsv = 1)[1].S[1]
    else
        L = opnorm(x)^2
    end

    L *= lossfun.lipschitz

    primals = Float64[]
    duals = Float64[]
    gaps = Float64[]

    times = Float64[]
    t0 = time()

    it = 0

    while it < maxit
        it += 1

        primal = loss(lossfun, η, y) + λ * norm(coef, 1)

        θ = _dual_point(lossfun, η, y, fit_intercept)
        dual_scale = _dual_scale(x' * θ, λ)
        θ ./= dual_scale
        dua = dual(lossfun, θ, y)
        gap = primal - dua

        push!(primals, primal)
        push!(duals, dua)
        push!(times, time() - t0)

        rel_gap = gap / max(abs(primal), 1.0e-15)

        push!(gaps, gap)

        if rel_gap <= tol
            break
        end

        r = residual(lossfun, η, y)

        old_intercept = intercept

        grad = x' * r
        coef = st.(coef - grad / L, λ / L)
        η = x * coef .+ old_intercept

        intercept = update_intercept(intercept_strategy, lossfun, old_intercept, η, y)

        η .+= intercept - old_intercept
    end

    intercept_rescaled, coef_rescaled = rescalecoefs(
        coef,
        intercept,
        x_centers,
        x_scales;
        fit_intercept = fit_intercept,
    )

    return (
        intercept = intercept_rescaled,
        coef = coef_rescaled,
        primals = primals,
        duals = duals,
        gaps = gaps,
        time = times,
        passes = it,
    )
end
