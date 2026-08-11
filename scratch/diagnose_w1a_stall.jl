# Instrument the exact-strategy stall on w1a + cyclic CD.
#
# Runs an exact-strategy cyclic CD pass-by-pass with per-pass diagnostics:
#   - intercept value, |∂₀F| before and after the end-of-pass exact solve
#   - line-search step size α used after each pass
#   - within-pass intercept-gradient drift accumulated over the p coordinate
#     updates (signed sum of H_{0j}·Δβ_j across the pass)
#   - column-mean coherence: do the H_{0j} share sign? what is their
#     concentration?
#   - per-pass primal change
#
# Then runs the same configuration on news20-3pct (same severe imbalance,
# different design — exact strategy converges there per @fig-real-logreg) for
# contrast.

using Intercepts
using LinearAlgebra, Statistics, SparseArrays, Random
using LIBSVMdata
using Printf

function load_problem(
    name::String;
    n_positive::Union{Nothing, Int} = nothing,
    n_negative::Union{Nothing, Int} = nothing,
    min_nnz_per_column::Int = 1,
    seed::Int = 42,
)
    X, y = load_dataset(name, verbose = false)
    pos = maximum(y)
    y = Int.(y .== pos)

    if n_positive !== nothing || n_negative !== nothing
        rng = MersenneTwister(seed)
        pos_idx = findall(==(1), y)
        neg_idx = findall(==(0), y)
        n_pos_keep = n_positive === nothing ?
            length(pos_idx) :
            min(n_positive, length(pos_idx))
        n_neg_keep = n_negative === nothing ?
            length(neg_idx) :
            min(n_negative, length(neg_idx))
        keep_rows = sort(vcat(
            shuffle(rng, pos_idx)[1:n_pos_keep],
            shuffle(rng, neg_idx)[1:n_neg_keep],
        ))
        X = X[keep_rows, :]
        y = y[keep_rows]
        if issparse(X)
            Xc = X::SparseMatrixCSC
            keep = [
                Xc.colptr[j + 1] - Xc.colptr[j] >= min_nnz_per_column
                for j in 1:size(Xc, 2)
            ]
        else
            keep = [
                !iszero(maximum(@view X[:, j]) - minimum(@view X[:, j]))
                for j in 1:size(X, 2)
            ]
        end
        X = X[:, keep]
    end

    return X, y
end

# Copy of cdsolver with per-pass instrumentation. Strategy = :exact; randomize
# selectable. We expose:
#   intercept_pre, intercept_post : intercept value before / after exact solve
#   grad0_pre, grad0_post         : sum gradient on intercept axis pre/post
#   drift_pass                    : signed within-pass drift accumulated as
#                                   sum over j of H_{0j} · Δβ_j_std evaluated
#                                   at the pre-pass weights
#   alpha_pass                    : post-pass backtracking step finally accepted
#   primal_pre, primal_post       : primal before pass / after pass+intercept
import Intercepts:
    LogisticLoss,
    residual,
    gradient,
    hessian,
    loss,
    lambdamax,
    normalizefeatures,
    st,
    update_intercept_with_count,
    ExactStrategy,
    rescalecoefs

function run_instrumented(
    X,
    y,
    reg;
    randomize::Bool,
    maxit::Int,
    maxtime::Float64,
    tol::Real = 1.0e-10,
)
    lossfun = LogisticLoss()
    intercept_strategy = ExactStrategy()
    n, p = size(X)
    x, x_centers, x_scales = normalizefeatures(X, :standardize)
    x_sparse_offset = vec(x_centers ./ x_scales)
    sparse_norm = issparse(x)
    λmax = lambdamax(lossfun, x, y)
    λ = reg * λmax

    intercept = 0.0
    coef = zeros(p)
    η = zeros(n)

    # diagnostics
    intercept_history = Float64[]
    grad0_pre_history = Float64[]
    grad0_post_history = Float64[]
    drift_pass_history = Float64[]
    alpha_pass_history = Float64[]
    primal_pre_history = Float64[]
    primal_post_history = Float64[]
    relgap_history = Float64[]
    k0_history = Int[]

    # column-level coherence at the cold start (computed once, since w1a is
    # essentially symmetric across passes early on; we'll record it for the
    # report rather than per-pass)
    t0 = time()
    it = 0
    while it < maxit && (time() - t0) < maxtime
        it += 1

        # primal + dual at the start of the pass
        primal = loss(lossfun, η, y) + λ * norm(coef, 1)
        r = residual(lossfun, η, y)
        θ = r .- mean(r)
        grad_all = vec(x' * θ)
        if sparse_norm
            θsum = sum(θ)
            grad_all .-= x_sparse_offset .* θsum
        end
        dual_scale = max(1, norm(grad_all, Inf) / λ)
        θ ./= dual_scale
        dua = Intercepts.dual(lossfun, θ, y)
        gap = primal - dua
        relgap = gap / max(abs(primal), 1.0e-10)
        push!(primal_pre_history, primal)
        push!(relgap_history, relgap)
        push!(intercept_history, intercept)

        # intercept-axis gradient before the pass
        η_grad_pre = gradient(lossfun, η, y)
        grad0_pre = sum(η_grad_pre)
        push!(grad0_pre_history, grad0_pre)

        # weights at the START of the pass: used to compute the "frozen-Hessian"
        # within-pass drift later.
        w_start = hessian(lossfun, η, y)
        H0j_start = Vector{Float64}(undef, p)
        for j in 1:p
            xc = view(x, :, j)
            H0j_start[j] = dot(xc, w_start)
            if sparse_norm
                H0j_start[j] -= x_sparse_offset[j] * sum(w_start)
            end
        end

        relgap <= tol && break

        ind = randomize ? randperm(p) : (1:p)
        coef_old = copy(coef)
        intercept_old = intercept

        # within-pass coordinate-update accumulator (standardized scale)
        deltas = zeros(p)

        for (inner_it, j) in enumerate(ind)
            coef_j = coef[j]
            η_grad = gradient(lossfun, η, y)
            η_hess = hessian(lossfun, η, y)
            grad = dot(view(x, :, j), η_grad)
            hess = dot(view(x, :, j), view(x, :, j) .* η_hess)
            if sparse_norm
                grad -= x_sparse_offset[j] * sum(η_grad)
                hess -= 2 * x_sparse_offset[j] * dot(η_hess, view(x, :, j)) -
                    x_sparse_offset[j]^2 * sum(η_hess)
            end
            if hess == 0
                hess = 1.0e-10
            end
            coef[j] = st(coef_j - grad / hess, λ / hess)
            diff = coef_j - coef[j]
            deltas[j] = -diff # positive if coef increased
            if diff != 0
                η .-= diff .* view(x, :, j)
                if sparse_norm
                    η .+= diff * x_sparse_offset[j]
                end
            end

            # NOTE: with update_freq=1, the in-pass intercept update fires only
            # at inner_it == p. Here we keep that semantics (no inner intercept
            # update except at the end) — but actually for :exact the strategy
            # is still applied once per pass via the loop. Replicate cdsolver:
            update_when = floor(p / 1)
            if inner_it % update_when == 0
                intercept_prev = intercept
                intercept, k0 = update_intercept_with_count(
                    intercept_strategy,
                    lossfun,
                    intercept_prev,
                    η,
                    y,
                )
                η .+= intercept - intercept_prev
                push!(k0_history, k0)
            end
        end

        # within-pass drift in intercept gradient, evaluated under the
        # start-of-pass weights (so this is the "linearized" drift that
        # @prp-exact-cost discusses): ∑_j H_{0j} Δβ_j.
        drift = dot(H0j_start, deltas)
        push!(drift_pass_history, drift)

        # intercept-axis gradient AFTER the exact solve (should be ≈ 0)
        η_grad_post = gradient(lossfun, η, y)
        grad0_post = sum(η_grad_post)
        push!(grad0_post_history, grad0_post)

        # post-pass primal + line search
        primal_after = loss(lossfun, η, y) + λ * norm(coef, 1)
        coef_diff = coef - coef_old
        intercept_diff = intercept - intercept_old
        alpha = 1.0
        while primal_after > primal && alpha > 1.0e-4
            alpha *= 0.1
            coef = coef_old + alpha * coef_diff
            intercept = intercept_old + alpha * intercept_diff
            η = x * coef .+ intercept
            if sparse_norm
                η .-= dot(x_sparse_offset, coef)
            end
            primal_after = loss(lossfun, η, y) + λ * norm(coef, 1)
        end
        push!(alpha_pass_history, alpha)
        push!(primal_post_history, primal_after)
    end

    return (; intercept_history, grad0_pre_history, grad0_post_history,
              drift_pass_history, alpha_pass_history,
              primal_pre_history, primal_post_history, relgap_history,
              k0_history)
end

# ---------- run on w1a (the headline stall) ----------
println("Loading w1a ...")
X_w1a, y_w1a = load_problem("w1a")
n, p = size(X_w1a)
@printf(
    "w1a: n=%d, p=%d, %% positive = %.3f, density = %.4f\n",
    n,
    p,
    mean(y_w1a),
    nnz(X_w1a) / (n * p),
)

println("\n=== w1a + cyclic + exact ===")
res_w1a_cyc = run_instrumented(
    X_w1a,
    y_w1a,
    0.05;
    randomize = false,
    maxit = 60,
    maxtime = 60.0,
)
@printf("passes run: %d\n", length(res_w1a_cyc.relgap_history))
@printf("final relgap: %.3e\n", res_w1a_cyc.relgap_history[end])
println("\n  pass    relgap     grad0_pre    grad0_post  drift_acc   α      Δprimal")
for k in 1:length(res_w1a_cyc.relgap_history)
    rg = res_w1a_cyc.relgap_history[k]
    g0p = k <= length(res_w1a_cyc.grad0_pre_history) ?
        res_w1a_cyc.grad0_pre_history[k] :
        NaN
    g0q = k <= length(res_w1a_cyc.grad0_post_history) ?
        res_w1a_cyc.grad0_post_history[k] :
        NaN
    dr = k <= length(res_w1a_cyc.drift_pass_history) ?
        res_w1a_cyc.drift_pass_history[k] :
        NaN
    α = k <= length(res_w1a_cyc.alpha_pass_history) ?
        res_w1a_cyc.alpha_pass_history[k] :
        NaN
    dp = (k <= length(res_w1a_cyc.primal_post_history)) ?
        (res_w1a_cyc.primal_post_history[k] - res_w1a_cyc.primal_pre_history[k]) :
        NaN
    @printf("  %3d  %.3e  %+.3e   %+.3e  %+.3e  %.0e   %+.3e\n", k, rg, g0p, g0q, dr, α, dp)
end

println("\n=== w1a + permuted + exact ===")
Random.seed!(42)
res_w1a_perm = run_instrumented(
    X_w1a,
    y_w1a,
    0.05;
    randomize = true,
    maxit = 60,
    maxtime = 60.0,
)
@printf("passes run: %d\n", length(res_w1a_perm.relgap_history))
@printf("final relgap: %.3e\n", res_w1a_perm.relgap_history[end])
for k in 1:length(res_w1a_perm.relgap_history)
    rg = res_w1a_perm.relgap_history[k]
    α = k <= length(res_w1a_perm.alpha_pass_history) ?
        res_w1a_perm.alpha_pass_history[k] :
        NaN
    dr = k <= length(res_w1a_perm.drift_pass_history) ?
        res_w1a_perm.drift_pass_history[k] :
        NaN
    @printf("  %3d  relgap=%.3e  α=%.0e  drift=%+.3e\n", k, rg, α, dr)
end
