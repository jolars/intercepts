# Reproduce the w1a + cyclic + exact stall on a constructed design.
#
# Hypothesis: the stall is driven by
#   (i)   severe class imbalance (rare positive class)
#   (ii)  sparse 0/1 features biased toward the positive class
#   (iii) cyclic ordering that visits these features in the same order each pass
#   (iv)  the exact strategy's large end-of-pass intercept move
#
# (i)–(iv) combine to make the cyclic-CD pass direction non-descent. The
# post-pass primal line search shrinks step size to ~1e-4, then exits without
# resetting α = 0, so the next pass restarts from essentially the same point
# and the iteration stalls.
#
# This script tries the constructed design and reports whether the stall
# reproduces.

using Intercepts
using Random, LinearAlgebra, Statistics, SparseArrays, Printf
using LIBSVMdata
import Intercepts: LogisticLoss, residual, gradient, hessian, loss, lambdamax,
                   normalizefeatures, st, update_intercept_with_count,
                   ExactStrategy

function run_short(X, y, reg; randomize::Bool, maxit::Int = 60,
                   maxtime::Float64 = 60.0)
    lossfun = LogisticLoss()
    intercept_strategy = ExactStrategy()
    n, p = size(X)
    x, x_centers, x_scales = normalizefeatures(X, :standardize)
    x_sparse_offset = vec(x_centers ./ x_scales)
    sparse_norm = issparse(x)
    λmax = lambdamax(lossfun, x, y); λ = reg * λmax

    intercept = 0.0; coef = zeros(p); η = zeros(n)

    relgap_history = Float64[]
    alpha_history = Float64[]
    drift_history = Float64[]
    primal_pre_history = Float64[]
    primal_post_history = Float64[]

    t0 = time(); it = 0
    while it < maxit && (time() - t0) < maxtime
        it += 1
        primal = loss(lossfun, η, y) + λ * norm(coef, 1)
        r = residual(lossfun, η, y); θ = r .- mean(r)
        grad_all = vec(x' * θ)
        if sparse_norm
            grad_all .-= x_sparse_offset .* sum(θ)
        end
        dual_scale = max(1, norm(grad_all, Inf) / λ); θ ./= dual_scale
        dua = Intercepts.dual(lossfun, θ, y); gap = primal - dua
        relgap = gap / max(abs(primal), 1e-10)
        push!(relgap_history, relgap); push!(primal_pre_history, primal)

        w_start = hessian(lossfun, η, y)
        H0j_start = Vector{Float64}(undef, p)
        for j = 1:p
            xc = view(x, :, j)
            H0j_start[j] = dot(xc, w_start)
            if sparse_norm
                H0j_start[j] -= x_sparse_offset[j] * sum(w_start)
            end
        end

        relgap <= 1e-10 && break

        ind = randomize ? randperm(p) : (1:p)
        coef_old = copy(coef); intercept_old = intercept
        deltas = zeros(p)

        for (inner_it, j) in enumerate(ind)
            coef_j = coef[j]
            η_grad = gradient(lossfun, η, y); η_hess = hessian(lossfun, η, y)
            grad = dot(view(x,:,j), η_grad)
            hess = dot(view(x,:,j), view(x,:,j) .* η_hess)
            if sparse_norm
                grad -= x_sparse_offset[j] * sum(η_grad)
                hess -= 2 * x_sparse_offset[j] * dot(η_hess, view(x,:,j)) -
                        x_sparse_offset[j]^2 * sum(η_hess)
            end
            hess == 0 && (hess = 1e-10)
            coef[j] = st(coef_j - grad/hess, λ/hess)
            diff = coef_j - coef[j]
            deltas[j] = -diff
            if diff != 0
                η .-= diff .* view(x,:,j)
                if sparse_norm
                    η .+= diff * x_sparse_offset[j]
                end
            end
            if inner_it == p
                intercept_prev = intercept
                intercept, _ = update_intercept_with_count(intercept_strategy,
                                                          lossfun, intercept_prev, η, y)
                η .+= intercept - intercept_prev
            end
        end

        drift = dot(H0j_start, deltas)
        push!(drift_history, drift)

        primal_after = loss(lossfun, η, y) + λ * norm(coef, 1)
        coef_diff = coef - coef_old; intercept_diff = intercept - intercept_old
        alpha = 1.0
        while primal_after > primal && alpha > 1e-4
            alpha *= 0.1
            coef = coef_old + alpha * coef_diff
            intercept = intercept_old + alpha * intercept_diff
            η = x * coef .+ intercept
            if sparse_norm
                η .-= dot(x_sparse_offset, coef)
            end
            primal_after = loss(lossfun, η, y) + λ * norm(coef, 1)
        end
        push!(alpha_history, alpha)
        push!(primal_post_history, primal_after)
    end
    return (; relgap_history, alpha_history, drift_history)
end

function summarize(label, res)
    n_runs = length(res.relgap_history)
    n_low_alpha = count(<(1e-3), res.alpha_history)
    @printf("\n[%s] passes=%d final_relgap=%.3e n_passes_alpha<1e-3=%d\n",
            label, n_runs, res.relgap_history[end], n_low_alpha)
    @printf("[%s] final drift sample: %s\n", label,
            join((@sprintf("%+.2e", d) for d in res.drift_history[max(1, end-4):end]), ", "))
    @printf("[%s] final alpha sample: %s\n", label,
            join((@sprintf("%.0e", a) for a in res.alpha_history[max(1, end-4):end]), ", "))
end

# ---------- 1. Constructed sparse-binary, severe imbalance ----------
# Match w1a's basic shape: n=2477, p=300, density ≈ 0.04, μ0 = 0.029.
# β has s=10 nonzero entries → predictive sparse features (some columns
# correlate with y). Default `means = nothing` for :binary, so x is iid binary
# bits. The Bernoulli sampling at sigmoid(η + β0) generates y.
Random.seed!(2025)
X_c, y_c = Intercepts.generatedata(2477, 300; response = :binomial,
                                    x_type = :binary, x_density = 0.04,
                                    μ0 = 0.029, s = 10, amplitude = 1.0)
@printf("Constructed: n=%d, p=%d, %%pos=%.3f, density=%.4f\n",
        size(X_c)..., mean(y_c), nnz(X_c) / prod(size(X_c)))

println("=== Constructed binary + cyclic + exact ===")
res_c = run_short(X_c, y_c, 0.05; randomize = false, maxit = 100, maxtime = 120.0)
summarize("constructed cyclic", res_c)

# ---------- 2. Constructed sparse-binary at moderate imbalance ----------
Random.seed!(2025)
X_m, y_m = Intercepts.generatedata(2477, 300; response = :binomial,
                                    x_type = :binary, x_density = 0.04,
                                    μ0 = 0.30, s = 10, amplitude = 1.0)
@printf("\nModerate-imb: %%pos=%.3f\n", mean(y_m))
println("=== Constructed binary moderate + cyclic + exact ===")
res_m = run_short(X_m, y_m, 0.05; randomize = false, maxit = 100, maxtime = 60.0)
summarize("moderate cyclic", res_m)

# ---------- 3. Same constructed severe-imb under PERMUTED ordering ----------
println("\n=== Constructed binary severe + permuted + exact ===")
Random.seed!(7)
res_cp = run_short(X_c, y_c, 0.05; randomize = true, maxit = 200, maxtime = 60.0)
summarize("constructed permuted", res_cp)

# ---------- 4. Diagnostic: drift coherence on w1a, constructed, news20 ----------
function design_coherence(X, y; n_warm::Int = 3, reg::Float64 = 0.05)
    # warm up with a few exact-strategy cyclic passes, then look at the
    # signed-drift contribution per column: H_{0j} after warmup × Δβ_j sign
    # pattern from a single subsequent pass. The "coherence" is the ratio
    # |Σ H_{0j} Δβ_j| / Σ |H_{0j} Δβ_j| — close to 1 means strongly aligned.
    lossfun = LogisticLoss()
    n, p = size(X)
    x, x_centers, x_scales = normalizefeatures(X, :standardize)
    x_sparse_offset = vec(x_centers ./ x_scales)
    sparse_norm = issparse(x)
    λmax = lambdamax(lossfun, x, y); λ = reg * λmax
    intercept = 0.0; coef = zeros(p); η = zeros(n)
    for _ = 1:n_warm
        coef_old = copy(coef); intercept_old = intercept
        for j = 1:p
            coef_j = coef[j]
            η_grad = gradient(lossfun, η, y); η_hess = hessian(lossfun, η, y)
            g = dot(view(x,:,j), η_grad)
            h = dot(view(x,:,j), view(x,:,j) .* η_hess)
            if sparse_norm
                g -= x_sparse_offset[j] * sum(η_grad)
                h -= 2 * x_sparse_offset[j] * dot(η_hess, view(x,:,j)) -
                     x_sparse_offset[j]^2 * sum(η_hess)
            end
            h == 0 && (h = 1e-10)
            coef[j] = st(coef_j - g/h, λ/h)
            diff = coef_j - coef[j]
            if diff != 0
                η .-= diff .* view(x,:,j)
                if sparse_norm
                    η .+= diff * x_sparse_offset[j]
                end
            end
        end
        intercept_prev = intercept
        intercept, _ = update_intercept_with_count(ExactStrategy(),
                                                  lossfun, intercept_prev, η, y)
        η .+= intercept - intercept_prev
    end

    # Now collect H_{0j} and a one-pass Δβ_j (under the same η)
    w = hessian(lossfun, η, y)
    H0j = Vector{Float64}(undef, p)
    for j = 1:p
        xc = view(x,:,j)
        H0j[j] = dot(xc, w)
        if sparse_norm
            H0j[j] -= x_sparse_offset[j] * sum(w)
        end
    end
    deltas = zeros(p)
    eta_copy = copy(η); coef_copy = copy(coef)
    for j = 1:p
        coef_j = coef_copy[j]
        η_grad = gradient(lossfun, eta_copy, y); η_hess = hessian(lossfun, eta_copy, y)
        g = dot(view(x,:,j), η_grad)
        h = dot(view(x,:,j), view(x,:,j) .* η_hess)
        if sparse_norm
            g -= x_sparse_offset[j] * sum(η_grad)
            h -= 2 * x_sparse_offset[j] * dot(η_hess, view(x,:,j)) -
                 x_sparse_offset[j]^2 * sum(η_hess)
        end
        h == 0 && (h = 1e-10)
        coef_copy[j] = st(coef_j - g/h, λ/h)
        diff = coef_j - coef_copy[j]
        deltas[j] = -diff
        if diff != 0
            eta_copy .-= diff .* view(x,:,j)
            if sparse_norm
                eta_copy .+= diff * x_sparse_offset[j]
            end
        end
    end
    contribs = H0j .* deltas
    num = abs(sum(contribs)); den = sum(abs, contribs)
    sign_pos = count(>(0), H0j) / p
    return (; coherence = num / max(den, 1e-12),
              drift_sum = sum(contribs),
              drift_abs = sum(abs, contribs),
              frac_H0_pos = sign_pos,
              n_active = count(!=(0), deltas),
              μ_y = mean(y))
end

println("\n--- coherence diagnostics ---")
println("(coherence = |Σ H_{0j} Δβ_j| / Σ |H_{0j} Δβ_j|; 1.0 = perfectly aligned)")

X_w, y_w = let
    Xw, yw = LIBSVMdata.load_dataset("w1a", verbose = false)
    pos = maximum(yw); yw = Int.(yw .== pos)
    Xw, yw
end
@printf("w1a            : %s\n", design_coherence(X_w, y_w))
@printf("constructed sev: %s\n", design_coherence(X_c, y_c))
@printf("constructed mod: %s\n", design_coherence(X_m, y_m))

# also dense gaussian (heatmap-style) for contrast
X_g, y_g = Intercepts.generatedata(500, 1000; response = :binomial,
                                    x_type = :normal, μ0 = 0.99, s = 10, ρ = 0.6,
                                    means = :random, amplitude = 1.0)
@printf("gaussian dense (heatmap μ0=0.99): %s\n", design_coherence(X_g, y_g))
