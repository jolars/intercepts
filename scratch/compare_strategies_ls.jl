# Does the post-pass line-search trap fire on w1a + cyclic with the NEWTON
# (single step) and GRADIENT strategies, or is it specific to EXACT?

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
    NewtonStrategy,
    GradientStrategy,
    ExactStrategy

function run_strat(
    X,
    y,
    reg;
    strategy,
    randomize::Bool,
    maxit::Int = 200,
    maxtime::Float64 = 30.0,
)
    lossfun = LogisticLoss()
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
    alpha_history = Float64[]
    t0 = time()
    for it in 1:maxit
        (time() - t0) >= maxtime && break
        primal = loss(lossfun, η, y) + λ * norm(coef, 1)
        r = residual(lossfun, η, y)
        θ = r .- mean(r)
        grad_all = vec(x' * θ)
        if sparse_norm
            grad_all .-= x_sparse_offset .* sum(θ)
        end
        ds = max(1, norm(grad_all, Inf) / λ)
        θ ./= ds
        dua = Intercepts.dual(lossfun, θ, y)
        gap = primal - dua
        relgap = gap / max(abs(primal), 1.0e-10)
        push!(relgap_history, relgap)
        relgap <= 1.0e-10 && break
        ind = randomize ? randperm(p) : (1:p)
        coef_old = copy(coef)
        intercept_old = intercept
        for (inner_it, j) in enumerate(ind)
            coef_j = coef[j]
            ηg = gradient(lossfun, η, y)
            ηh = hessian(lossfun, η, y)
            g = dot(view(x, :, j), ηg)
            h = dot(view(x, :, j), view(x, :, j) .* ηh)
            if sparse_norm
                g -= x_sparse_offset[j] * sum(ηg)
                h -= 2 * x_sparse_offset[j] * dot(ηh, view(x, :, j)) -
                    x_sparse_offset[j]^2 * sum(ηh)
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
                intercept, _ = update_intercept_with_count(strategy, lossfun, ip, η, y)
                η .+= intercept - ip
            end
        end
        primal_after = loss(lossfun, η, y) + λ * norm(coef, 1)
        cdiff = coef - coef_old
        idiff = intercept - intercept_old
        α = 1.0
        while primal_after > primal && α > 1.0e-4
            α *= 0.1
            coef = coef_old + α * cdiff
            intercept = intercept_old + α * idiff
            η = x * coef .+ intercept
            if sparse_norm
                η .-= dot(x_sparse_offset, coef)
            end
            primal_after = loss(lossfun, η, y) + λ * norm(coef, 1)
        end
        push!(alpha_history, α)
    end
    return (; relgap_history, alpha_history)
end

Xw, yw = LIBSVMdata.load_dataset("w1a", verbose = false)
pos = maximum(yw)
yw = Int.(yw .== pos)

for (name, strat) in [
        ("newton", NewtonStrategy()),
        ("gradient", GradientStrategy()),
        ("exact", ExactStrategy()),
    ]
    res = run_strat(
        Xw,
        yw,
        0.05;
        strategy = strat,
        randomize = false,
        maxit = 1000,
        maxtime = 30.0,
    )
    n_low = count(<(1.0e-3), res.alpha_history)
    @printf(
        "strategy=%-8s passes=%4d  final_relgap=%.3e  passes_with_α<1e-3=%d\n",
        name,
        length(res.relgap_history),
        res.relgap_history[end],
        n_low,
    )
    if n_low > 0
        # find first low-alpha pass
        k = findfirst(<(1.0e-3), res.alpha_history)
        @printf("  first low-α at pass %d; relgap there %.3e\n", k, res.relgap_history[k])
    end
end
