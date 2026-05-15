using Intercepts
using Random
using GLM
using DrWatson
using ProjectRoot
using DataFrames
using JLD2
using Statistics
using LinearAlgebra

Random.seed!(42);

param_dict = Dict{String,Any}(
    "dataset" => ["w1a"],
    "reg" => [0.05],
    "strategy" => [:gradient, :newton, :exact],
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

outfile = @projectroot("results", "first-example.jld2");

@save outfile results


