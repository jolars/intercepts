using LinearAlgebra
using Statistics
using Random

function cdsolver(
    x::AbstractMatrix,
    y::AbstractVector,
    reg::Real = 0.1;
    lossfun::LossFunction = QuadraticLoss(),
    intercept_strategy::InterceptStrategy = GradientStrategy(),
    update_freq::Int = 1,
    tol::Real = 1e-10,
    maxit::Int = 1000,
    maxtime::Real = Inf,
    randomize::Bool = true,
    normalization::Symbol = :standardize,
    save_history::Bool = false,
    coef_init::Union{Nothing,AbstractVector} = nothing,
    intercept_init::Real = 0.0,
)
    n, p = size(x)

    y = Float64.(y)

    validateresponse(lossfun, y)

    x, x_centers, x_scales = normalizefeatures(x, normalization)

    x_sparse_offset = x_centers ./ x_scales

    sparse_norm = issparse(x) && normalization != :none

    fit_intercept = !(intercept_strategy isa NoIntercept)

    update_when = floor(size(x, 2) / update_freq)

    λmax = lambdamax(lossfun, x, y)
    λ = reg * λmax

    if coef_init === nothing
        intercept = 0.0
        coef = zeros(p)
        η = zeros(n)
    else
        length(coef_init) == p ||
            throw(ArgumentError("coef_init must have length p = $p"))
        coef_user = Float64.(coef_init)
        coef = coef_user .* vec(x_scales)
        intercept =
            fit_intercept ? Float64(intercept_init) + dot(vec(x_centers), coef_user) : 0.0
        η = x * coef .+ intercept
        if sparse_norm
            η .-= dot(x_sparse_offset, coef)
        end
    end
    r = residual(lossfun, η, y)

    intercepts = Vector{Float64}(undef, 0)
    coefs = Vector{Vector{Float64}}(undef, 0)

    primals = Float64[]
    duals = Float64[]
    gaps = Float64[]
    relgaps = Float64[]

    times = Float64[]
    inner_steps = Int[]
    t0 = time()

    # Tseng--Yun proximal-Newton CD: each coord step is wrapped in an Armijo
    # backtrack so the iterate is descent by construction.
    armijo_c = 1.0e-4
    armijo_shrink = 0.5
    armijo_max_backtracks = 30

    η_trial = similar(η)

    it = 0

    while it < maxit && (time() - t0) < maxtime
        it += 1

        primal = loss(lossfun, η, y) + λ * norm(coef, 1)

        r = residual(lossfun, η, y)
        θ = r

        if fit_intercept
            θ .-= mean(r)
        end

        grad = x' * θ

        if sparse_norm
            θsum = sum(θ)
            for j = 1:p
                grad[j] -= x_sparse_offset[j] * θsum
            end
        end

        dual_scale = max(1, norm(grad, Inf) / λ)

        θ ./= dual_scale
        dua = dual(lossfun, θ, y)
        gap = primal - dua

        push!(primals, primal)
        push!(duals, dua)
        push!(times, time() - t0)

        intercept_rescaled, coef_rescaled = rescalecoefs(
            coef,
            intercept,
            x_centers,
            x_scales;
            fit_intercept = fit_intercept,
        )

        if save_history
            push!(intercepts, intercept_rescaled)
            push!(coefs, coef_rescaled)
        end

        rel_gap = gap / max(abs(primal), 1e-15)

        push!(relgaps, gap / max(abs(primal), 1e-10))
        push!(gaps, gap)

        if rel_gap <= tol
            break
        end

        ind = if randomize
            randperm(p)
        else
            1:p
        end

        # Cached loss(η). Refreshed on every accepted coord step and after each
        # in-pass intercept update.
        loss_eta = loss(lossfun, η, y)

        pass_inner_steps = 0

        for (inner_it, j) in enumerate(ind)
            coef_j = coef[j]

            η_grad = gradient(lossfun, η, y)
            η_hess = hessian(lossfun, η, y)

            xj = x[:, j]

            grad_j = dot(xj, η_grad)
            hess_j = dot(xj, xj .* η_hess)

            if sparse_norm
                grad_j -= x_sparse_offset[j] * sum(η_grad)
                hess_j -=
                    2 * x_sparse_offset[j] * dot(η_hess, xj) -
                    x_sparse_offset[j]^2 * sum(η_hess)
            end

            if hess_j <= 0
                hess_j = 1.0e-10
            end

            β_new = st(coef_j - grad_j / hess_j, λ / hess_j)
            d_j = β_new - coef_j

            if d_j != 0
                # Tseng--Yun model directional derivative; ≤ 0 at the prox optimum.
                Δmodel = grad_j * d_j + λ * (abs(β_new) - abs(coef_j))

                offset_j = sparse_norm ? x_sparse_offset[j] : 0.0
                α = 1.0
                accepted_α = 0.0
                accepted_loss = loss_eta
                for _ = 0:armijo_max_backtracks
                    factor = α * d_j
                    copyto!(η_trial, η)
                    if sparse_norm
                        η_trial .-= factor * offset_j
                    end
                    η_trial .+= factor .* xj

                    loss_trial = loss(lossfun, η_trial, y)
                    ΔF = (loss_trial - loss_eta) +
                         λ * (abs(coef_j + factor) - abs(coef_j))

                    # Tseng--Yun sufficient decrease + FP-precision tolerance
                    # for near-optimal iterates where loss_trial - loss_eta is
                    # cancellation-noise.
                    fp_tol = 4 * eps(Float64) * (1.0 + abs(loss_eta))
                    if ΔF <= armijo_c * α * Δmodel + fp_tol
                        accepted_α = α
                        accepted_loss = loss_trial
                        break
                    end
                    α *= armijo_shrink
                end

                if accepted_α > 0
                    coef[j] = accepted_α == 1.0 ? β_new : coef_j + accepted_α * d_j
                    copyto!(η, η_trial)
                    loss_eta = accepted_loss
                end
            end

            if inner_it % update_when == 0 && fit_intercept
                intercept_prev = intercept
                intercept, k0 = update_intercept_with_count(
                    intercept_strategy,
                    lossfun,
                    intercept_prev,
                    η,
                    y,
                )
                η .+= intercept - intercept_prev
                loss_eta = loss(lossfun, η, y)
                pass_inner_steps += k0
            end
        end

        push!(inner_steps, pass_inner_steps)
    end

    intercept_rescaled, coef_rescaled =
        rescalecoefs(coef, intercept, x_centers, x_scales; fit_intercept = fit_intercept)

    return (
        intercept = intercept_rescaled,
        coef = coef_rescaled,
        primals = primals,
        duals = duals,
        gaps = gaps,
        relgaps = relgaps,
        time = times,
        inner_steps = inner_steps,
        passes = it,
        coefs = save_history ? reduce(hcat, coefs) : Matrix{Float64}(undef, 0, 0),
        intercepts = intercepts,
        λ = λ,
        λmax = λmax,
    )
end
