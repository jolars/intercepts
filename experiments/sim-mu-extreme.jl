using Intercepts
using Random
using DrWatson
using ProjectRoot
using JLD2

Random.seed!(1234);

# Extreme-imbalance logistic regression: the gradient strategy applies only the
# fraction H_00/L_0 of the Newton intercept correction. For mu0 -> 1, H_00 -> 0
# while L_0 stays fixed, so the gradient strategy stalls.
param_dict = Dict{String, Any}(
    "it" => collect(1:1),
    "n" => [500],
    "p" => [1000],
    "s" => [10],
    "reg" => [0.05],
    "μ0" => [0.5, 0.9, 0.95, 0.99],
    "strategy" => [:gradient, :bt_gradient, :newton, :exact],
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
        randomize = false,
    )

    d_exp = Dict{String, Any}(copy(d))
    d_exp["time"] = res.time
    d_exp["gaps"] = res.gaps
    d_exp["relgaps"] = res.relgaps
    d_exp["primals"] = res.primals
    d_exp["duals"] = res.duals

    push!(results, d_exp)
end

suboptimality_against_shared_dual!(
    results;
    instance_of = r -> (r["it"], r["n"], r["p"], r["s"], r["reg"], r["μ0"]),
)

outfile = @projectroot("results", "sim-mu-extreme.jld2");

@save outfile results
