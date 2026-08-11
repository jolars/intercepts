using Intercepts
using Random
using DrWatson
using ProjectRoot
using JLD2

Random.seed!(1234);

# Cold-start cyclic CD on extreme-imbalance logistic regression.
# An unguarded Newton step sees a large initial |partial_0 F| at zero-init and
# could oscillate with cyclic ordering (Lemma 3.2 + Corollary: the Newton
# residual perturbs subsequent coefficient gradients); the per-coord Armijo
# inside NewtonStrategy absorbs the initial transient.
param_dict = Dict{String, Any}(
    "it" => collect(1:5),
    "n" => [500],
    "p" => [1000],
    "s" => [10],
    "reg" => [0.05],
    "μ0" => [0.5, 0.9, 0.95, 0.99],
    "strategy" => [:newton, :unguarded_newton, :exact],
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
        maxit = 1000,
    )

    d_exp = Dict{String, Any}(copy(d))
    d_exp["time"] = res.time
    d_exp["gaps"] = res.gaps
    d_exp["relgaps"] = res.relgaps
    d_exp["primals"] = res.primals

    push!(results, d_exp)
end

outfile = @projectroot("results", "sim-cold-start.jld2");

@save outfile results
