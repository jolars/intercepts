# What does the w1a + cyclic + exact iteration look like if we REMOVE the
# post-pass primal line search? Does it diverge, oscillate, or converge?
#
# This tests whether the line search itself is the trap (by exiting at
# α = 1e-4 along an ascent direction, freezing the iterate) or whether the
# underlying iteration is genuinely non-convergent.

using Intercepts
using Random, LinearAlgebra, Statistics, SparseArrays, Printf
using LIBSVMdata
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
    ExactStrategy

function run_no_ls(X, y, reg; randomize::Bool, maxit::Int = 200)
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
    relgap_history = Float64[]
    primal_history = Float64[]

    for it in 1:maxit
        primal = loss(lossfun, η, y) + λ * norm(coef, 1)
        r = residual(lossfun, η, y)
        θ = r .- mean(r)
        grad_all = vec(x' * θ)
        if sparse_norm
            grad_all .-= x_sparse_offset .* sum(θ)
        end
        dual_scale = max(1, norm(grad_all, Inf) / λ)
        θ ./= dual_scale
        dua = Intercepts.dual(lossfun, θ, y)
        gap = primal - dua
        relgap = gap / max(abs(primal), 1.0e-10)
        push!(relgap_history, relgap)
        push!(primal_history, primal)
        relgap <= 1.0e-10 && break
        ind = randomize ? randperm(p) : (1:p)
        for (inner_it, j) in enumerate(ind)
            coef_j = coef[j]
            η_grad = gradient(lossfun, η, y)
            η_hess = hessian(lossfun, η, y)
            g = dot(view(x, :, j), η_grad)
            h = dot(view(x, :, j), view(x, :, j) .* η_hess)
            if sparse_norm
                g -= x_sparse_offset[j] * sum(η_grad)
                h -= 2 * x_sparse_offset[j] * dot(η_hess, view(x, :, j)) -
                    x_sparse_offset[j]^2 * sum(η_hess)
            end
            h == 0 && (h = 1.0e-10)
            coef[j] = st(coef_j - g / h, λ / h)
            diff = coef_j - coef[j]
            if diff != 0
                η .-= diff .* view(x, :, j)
                if sparse_norm
                    η .+= diff * x_sparse_offset[j]
                end
            end
            if inner_it == p
                ip = intercept
                intercept, _ = update_intercept_with_count(
                    intercept_strategy,
                    lossfun,
                    ip,
                    η,
                    y,
                )
                η .+= intercept - ip
            end
        end
    end
    return (; relgap_history, primal_history)
end

Xw, yw = LIBSVMdata.load_dataset("w1a", verbose = false)
pos = maximum(yw)
yw = Int.(yw .== pos)

println("=== w1a + cyclic + exact, NO line search ===")
res = run_no_ls(Xw, yw, 0.05; randomize = false, maxit = 200)
@printf(
    "passes=%d  final relgap=%.3e  final primal=%.6f\n",
    length(res.relgap_history),
    res.relgap_history[end],
    res.primal_history[end],
)
@printf(
    "min relgap during run: %.3e at pass %d\n",
    minimum(res.relgap_history),
    argmin(res.relgap_history),
)
@printf("trajectory (every 10):\n")
for (k, rg) in enumerate(res.relgap_history)
    if k <= 12 || k == length(res.relgap_history) || k % 10 == 0
        @printf("  pass %3d  relgap=%.3e  primal=%.6f\n", k, rg, res.primal_history[k])
    end
end
