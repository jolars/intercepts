using Intercepts
using Random
using DrWatson
using ProjectRoot
using JLD2
using LinearAlgebra
using Statistics

# Per-cell intercept-Hessian measurement on the (μ₀, λ/λ_max, seed) grid that
# mirrors sim-mu-reg-heatmap.jl. Records H₀₀(η̂) = Σᵢ wᵢ(η̂) at the
# Newton-strategy converged iterate, on the same standardized logistic
# problems used by the heatmap. This feeds the iterate-level prediction
# L₀ / H₀₀(η̂) of @cor-rate-gap that intercepts.qmd compares against the
# empirical pass-count ratios in @fig-mu-reg-gradient.

param_dict = Dict{String,Any}(
    "n" => [500],
    "p" => [1000],
    "s" => [10],
    "ρ" => [0.6],
    "μ0" => [0.5, 0.7, 0.9, 0.95, 0.99],
    "reg" => [0.5, 0.2, 0.1, 0.05, 0.02, 0.01],
    "seed" => [1, 2, 3],
)

params = dict_list(param_dict)

const MAXIT = 5000
const TOL = 1.0e-12

results = Vector{Dict{String,Any}}()

for (i, d) in enumerate(params)
    @unpack n, p, s, ρ, μ0, reg, seed = d

    Random.seed!(seed)
    X, y = generatedata(
        n,
        p;
        response = :binomial,
        μ0 = μ0,
        x_type = :normal,
        x_density = 1.0,
        ρ = ρ,
        s = s,
        amplitude = 1.0,
        means = :random,
    )

    lossfun = LogisticLoss()
    L0 = lossfun.lipschitz * n  # n · sup f''  =  n/4 for logistic

    res = cdsolver(
        X,
        y,
        reg;
        lossfun = lossfun,
        intercept_strategy = NewtonStrategy(),
        maxit = MAXIT,
        randomize = false,
        tol = TOL,
    )

    η_hat = X * res.coef .+ res.intercept
    H00 = sum(weight(lossfun, η_hat))

    d_exp = Dict{String,Any}(copy(d))
    d_exp["H00"] = H00
    d_exp["L0"] = L0
    d_exp["L0_over_H00"] = L0 / H00
    d_exp["newton_passes"] = res.passes
    d_exp["final_relgap"] = max(res.relgaps[end], 1.0e-20)

    println(
        "[$i/$(length(params))] μ0=$μ0 reg=$reg seed=$seed " *
        "H00=$(round(H00; sigdigits = 4)) L0/H00=$(round(L0 / H00; sigdigits = 3)) " *
        "passes=$(res.passes)",
    )

    push!(results, d_exp)
end

outfile = @projectroot("results", "sim-h00-heatmap.jld2")
@save outfile results
println("Saved results to $outfile")
