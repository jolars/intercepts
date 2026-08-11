using Intercepts
using Random
using DrWatson
using ProjectRoot
using JLD2

# Cyclic vs permuted coordinate-descent ordering on the w1a problem of
# @fig-first-example. The cyclic ordering is the headline case where the
# exact strategy fails to converge (its iterates oscillate around the joint
# optimum because each pass relinearises around an intercept value stale by
# the time the within-pass coordinate drift comes back to it). The permuted
# ordering breaks the deterministic worst-case pairing and lets the exact
# strategy converge, matching the behavior of cyclic Newton. Promotes the
# prior footnote to a Results-section figure.

Random.seed!(42)

param_dict = Dict{String, Any}(
    "dataset" => ["w1a"],
    "reg" => [0.05],
    "strategy" => [:gradient, :newton, :exact],
    "randomize" => [false, true],
)

params = dict_list(param_dict)

results = []

for (i, d) in enumerate(params)
    @unpack dataset, reg, strategy, randomize = d

    res = real_experiment(
        dataset,
        strategy = strategy,
        reg = reg,
        response = :binomial,
        randomize = randomize,
    )

    d_exp = Dict{String, Any}(copy(d))
    d_exp["strategy"] = String(strategy)
    d_exp["ordering"] = randomize ? "permuted" : "cyclic"
    d_exp["time"] = res.time
    d_exp["gaps"] = res.gaps
    d_exp["relgaps"] = res.relgaps
    d_exp["primals"] = res.primals

    println(
        "[$i/$(length(params))] dataset=$dataset strategy=$strategy randomize=$randomize " *
            "passes=$(length(res.time)) final_relgap=$(res.relgaps[end])",
    )

    push!(results, d_exp)
end

outfile = @projectroot("results", "first-example-ordering.jld2")
@save outfile results
println("Saved results to $outfile")
