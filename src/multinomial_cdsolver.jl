using LinearAlgebra
using Random

# λmax for multinomial logistic with element-wise L1 on β[j, k].
# At β = 0, the optimal intercept fits class proportions:
#   β0[k] = log(n_k / n_K)  for k = 1..K-1
# giving uniform-across-i softmax probs p[k] = n_k / n. The KKT residual is
# X' R where R[i, k] = p[k] - 𝟙{y_i = k}; λmax is the max-abs entry.
function lambdamax_multinomial(
        f::MultinomialLogisticLoss,
        x::AbstractMatrix,
        y::AbstractVector{<:Integer},
    )
    n = size(x, 1)
    K = f.K
    counts = zeros(K)
    for yi in y
        counts[yi] += 1
    end
    counts = max.(counts, 0.5)
    p_marg = counts ./ sum(counts)

    R = Matrix{Float64}(undef, n, K - 1)
    @inbounds for i in 1:n, k in 1:(K - 1)
        R[i, k] = p_marg[k] - (y[i] == k ? 1.0 : 0.0)
    end
    return norm(x' * R, Inf)
end

function rescalecoefs_multinomial(
        coefs::AbstractMatrix,
        intercept::AbstractVector,
        centers::AbstractMatrix,
        scales::AbstractMatrix;
        fit_intercept::Bool = true,
    )
    p, Km1 = size(coefs)
    coefs_rescaled = copy(coefs)
    intercept_rescaled = copy(intercept)

    for k in 1:Km1
        shift = 0.0
        for j in 1:p
            coefs_rescaled[j, k] /= scales[j]
            shift += centers[j] * coefs_rescaled[j, k]
        end
        if fit_intercept
            intercept_rescaled[k] -= shift
        end
    end
    return intercept_rescaled, coefs_rescaled
end

"""
    multinomial_cdsolver(x, y, reg = 0.1; lossfun, kwargs...)

Fit a multinomial logistic model with an L1 penalty by proximal coordinate
descent. The final class is the reference class, so `coef` has size
`(p, K - 1)` and `intercept` has length `K - 1`. The result contains fitted
values, primal-dual diagnostics, `λ`, `λmax`, and optional histories. Iteration
stops at relative primal-dual gap `tol`, after `maxit` passes, or after
`maxtime` seconds.
"""
function multinomial_cdsolver(
        x::AbstractMatrix,
        y::AbstractVector{<:Integer},
        reg::Real = 0.1;
        lossfun::MultinomialLogisticLoss,
        intercept_strategy::InterceptStrategy = NewtonStrategy(),
        update_freq::Int = 1,
        tol::Real = 1.0e-10,
        maxit::Int = 1000,
        maxtime::Real = Inf,
        randomize::Bool = true,
        normalization::Symbol = :standardize,
        save_history::Bool = false,
    )
    n, p = size(x)
    K = lossfun.K
    Km1 = K - 1

    validateresponse(lossfun, y)

    x, x_centers, x_scales = normalizefeatures(x, normalization)

    fit_intercept = !(intercept_strategy isa NoIntercept)
    update_when = max(1, floor(Int, p / update_freq))

    λmax = lambdamax_multinomial(lossfun, x, y)
    λ = reg * λmax

    intercept = zeros(Km1)
    coef = zeros(p, Km1)
    η = zeros(n, Km1)

    primals = Float64[]
    duals = Float64[]
    gaps = Float64[]
    relgaps = Float64[]
    times = Float64[]
    intercepts_history = Vector{Vector{Float64}}(undef, 0)
    coefs_history = Vector{Matrix{Float64}}(undef, 0)
    t0 = time()

    # Tseng--Yun proximal-Newton CD: each (j, k) step is wrapped in an Armijo
    # backtrack on F so the iterate is descent by construction.
    armijo_c = 1.0e-4
    armijo_shrink = 0.5
    armijo_max_backtracks = 20

    η_trial = similar(η)

    it = 0
    while it < maxit && (time() - t0) < maxtime
        it += 1

        primal = loss(lossfun, η, y) + λ * sum(abs, coef)
        push!(primals, primal)
        push!(times, time() - t0)

        # The domain-safe centering step enforces intercept stationarity before
        # feature scaling enforces the penalized-coordinate constraint.
        Θ = _dual_point(lossfun, η, y, fit_intercept)
        grad_outer = x' * Θ
        dual_scale = _dual_scale(grad_outer, λ)
        Θ ./= dual_scale
        dua = dual(lossfun, Θ, y)
        gap = primal - dua
        relgap = gap / max(abs(primal), 1.0e-10)
        push!(duals, dua)
        push!(gaps, gap)
        push!(relgaps, relgap)

        if save_history
            intercept_resc, coef_resc = rescalecoefs_multinomial(
                coef,
                intercept,
                x_centers,
                x_scales;
                fit_intercept = fit_intercept,
            )
            push!(intercepts_history, intercept_resc)
            push!(coefs_history, coef_resc)
        end

        if relgap <= tol
            break
        end

        ind = randomize ? randperm(p) : (1:p)

        loss_eta = loss(lossfun, η, y)

        for (inner_it, j) in enumerate(ind)
            ps = softmax_probs(η) # Jacobi-within-j: probs frozen for all k of this feature

            for k in 1:Km1
                grad_jk = 0.0
                hess_jk = 0.0
                @inbounds for i in 1:n
                    diff_i = ps[i, k] - (y[i] == k ? 1.0 : 0.0)
                    xij = x[i, j]
                    grad_jk += xij * diff_i
                    pik = ps[i, k]
                    hess_jk += xij * xij * pik * (1 - pik)
                end

                if hess_jk <= 0
                    hess_jk = 1.0e-10
                end

                coef_jk = coef[j, k]
                β_new = st(coef_jk - grad_jk / hess_jk, λ / hess_jk)
                d_jk = β_new - coef_jk

                if d_jk == 0
                    continue
                end

                Δmodel = grad_jk * d_jk + λ * (abs(β_new) - abs(coef_jk))

                α = 1.0
                accepted_α = 0.0
                accepted_loss = loss_eta
                for _ in 0:armijo_max_backtracks
                    factor = α * d_jk
                    copyto!(η_trial, η)
                    @views η_trial[:, k] .+= factor .* x[:, j]

                    loss_trial = loss(lossfun, η_trial, y)
                    ΔF = (loss_trial - loss_eta) +
                        λ * (abs(coef_jk + factor) - abs(coef_jk))

                    fp_tol = 4 * eps(Float64) * (1.0 + abs(loss_eta))
                    if ΔF <= armijo_c * α * Δmodel + fp_tol
                        accepted_α = α
                        accepted_loss = loss_trial
                        break
                    end
                    α *= armijo_shrink
                end

                if accepted_α > 0
                    coef[j, k] = coef_jk + accepted_α * d_jk
                    copyto!(η, η_trial)
                    loss_eta = accepted_loss
                end
            end

            if inner_it % update_when == 0 && fit_intercept
                intercept_prev = copy(intercept)
                intercept = update_intercept(
                    intercept_strategy,
                    lossfun,
                    intercept_prev,
                    η,
                    y,
                )
                for k in 1:Km1
                    @views η[:, k] .+= intercept[k] - intercept_prev[k]
                end
                loss_eta = loss(lossfun, η, y)
            end
        end
    end

    intercept_out, coef_out = rescalecoefs_multinomial(
        coef,
        intercept,
        x_centers,
        x_scales;
        fit_intercept = fit_intercept,
    )

    return (
        intercept = intercept_out,
        coef = coef_out,
        primals = primals,
        duals = duals,
        gaps = gaps,
        relgaps = relgaps,
        time = times,
        passes = it,
        λ = λ,
        λmax = λmax,
        intercepts = intercepts_history,
        coefs = coefs_history,
    )
end
