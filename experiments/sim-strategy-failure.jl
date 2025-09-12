using Intercepts
using Random
using GLM
using DrWatson
using ProjectRoot
using DataFrames
using JLD2
using Statistics
using LinearAlgebra
using AlgebraOfGraphics
using CairoMakie

Random.seed!(42);

param_dict = Dict{String,Any}(
    "dataset" => ["w1a", "breast-cancer"],
    "reg" => [0.05],
    "strategy" => [:gradient, :newton, :convergence],
);

params = dict_list(param_dict);

results = [];

for (i, d) in enumerate(params)
    @unpack dataset, reg, strategy = d

    res = real_experiment(
        dataset,
        strategy = strategy,
        reg = reg,
        response = :binomial,
        randomize = false,
    )

    d_exp = Dict{String,Any}(copy(d))
    d_exp["time"] = res.time
    d_exp["gaps"] = res.gaps
    d_exp["relgaps"] = res.relgaps
    d_exp["primals"] = res.primals

    push!(results, d_exp)
end

df = DataFrame(results)
df = flatten(df, [:time, :relgaps, :gaps, :primals])

df.relgaps .= max.(df.relgaps, 1e-20)

spec = data(df) * mapping(:time, :relgaps, color = :strategy, col = :dataset)

layers = visual(Lines)

draw(
  layers * spec,
  axis=(yscale=log10,),
  facet = (; linkxaxes = :none),
  figure = (; size = (800, 300),)
)

