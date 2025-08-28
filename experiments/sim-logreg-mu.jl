using Intercepts
using Random
using GLM
using Statistics
using CairoMakie
using DrWatson
using ProjectRoot
using JLD2

Random.seed!(1234);

param_dict = Dict{String,Any}(
    "it" => collect(1:1),
    "n" => [500],
    "p" => [1000],
    "s" => [10],
    "reg" => [0.05],
    "μ0" => [0.5, 0.7, 0.9],
    "strategy" => [:gradient, :newton, :exact],
);

params = dict_list(param_dict);

results = [];

for (i, d) in enumerate(params)
    @unpack it, n, p, s, reg, μ0, strategy = d

    Random.seed!(it)

    res = simulated_experiment(
        n,
        p;
        response = :binomial,
        strategy = strategy,
        μ0 = μ0,
        s = s,
        reg = reg,
    )

    d_exp = Dict{String,Any}(copy(d))
    d_exp["time"] = res.time
    d_exp["gaps"] = res.gaps
    d_exp["relgaps"] = res.relgaps
    d_exp["primals"] = res.primals

    push!(results, d_exp)
end

outfile = @projectroot("results", "sim-logreg-mu.jld2");

@save outfile results
