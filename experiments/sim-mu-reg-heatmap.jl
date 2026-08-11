using Intercepts
using Random
using DrWatson
using ProjectRoot
using JLD2

# 2D sweep of class imbalance (μ₀) × regularization (λ/λ_max) for the three
# intercept strategies, on a fixed standardized imbalanced logistic design.
# Records pass counts to a fixed relative-gap tolerance per configuration so
# downstream code in intercepts.qmd can build the pass-count-ratio heatmap
# that tests the L₀/H₀₀ scaling prediction of @cor-rate-gap and
# @lem-gradient-partial across the (μ₀, λ) plane.

param_dict = Dict{String, Any}(
    "n" => [500],
    "p" => [1000],
    "s" => [10],
    "ρ" => [0.6],
    "μ0" => [0.5, 0.7, 0.9, 0.95, 0.99],
    "reg" => [0.5, 0.2, 0.1, 0.05, 0.02, 0.01],
    "strategy" => [:gradient, :newton, :exact],
    "seed" => [1, 2, 3],
)

params = dict_list(param_dict)

const MAXIT = 5000
const TOL = 1.0e-8

results = Vector{Dict{String, Any}}()

for (i, d) in enumerate(params)
    @unpack n, p, s, ρ, μ0, reg, strategy, seed = d

    Random.seed!(seed)

    res = simulated_experiment(
        n,
        p;
        response = :binomial,
        strategy = strategy,
        μ0 = μ0,
        s = s,
        reg = reg,
        ρ = ρ,
        randomize = false,
        maxit = MAXIT,
    )

    relgaps = res.relgaps
    # First index at which the relative gap drops below TOL, or maxit if not reached
    passes_to_tol = findfirst(g -> g <= TOL, relgaps)
    reached = passes_to_tol !== nothing
    pass_count = reached ? passes_to_tol : length(relgaps)
    final_relgap = max(relgaps[end], 1.0e-20)

    d_exp = Dict{String, Any}(copy(d))
    d_exp["strategy"] = String(strategy)
    d_exp["passes"] = pass_count
    d_exp["reached_tol"] = reached
    d_exp["final_relgap"] = final_relgap
    d_exp["wall_time"] = res.time[end]

    println(
        "[$i/$(length(params))] μ0=$μ0 reg=$reg strategy=$strategy seed=$seed " *
            "passes=$pass_count reached=$reached final_relgap=$final_relgap",
    )

    push!(results, d_exp)
end

outfile = @projectroot("results", "sim-mu-reg-heatmap.jld2")
@save outfile results
println("Saved results to $outfile")
