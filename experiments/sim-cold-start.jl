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

# Timing hygiene: the recorded per-pass wall-clock trajectory is a single
# time() sample per pass, so JIT compilation and GC pauses land directly on
# the plotted time axis. Warm up each strategy once to compile its solver
# specialization, collect garbage before every timed run, and keep the
# fastest of `nreps` identical runs (the solver is deterministic given the
# seed, so repetitions differ only in wall time).
for strategy in param_dict["strategy"]
    Random.seed!(0)
    simulated_experiment(
        50,
        20;
        response = :binomial,
        strategy = strategy,
        μ0 = 0.5,
        s = 5,
        reg = 0.05,
        randomize = false,
        maxit = 3,
    )
end

nreps = 3;

results = [];

for (i, d) in enumerate(params)
    @unpack it, n, p, s, reg, μ0, strategy = d

    res = nothing
    for rep in 1:nreps
        Random.seed!(it)
        GC.gc()

        r = simulated_experiment(
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

        if res === nothing || r.time[end] < res.time[end]
            res = r
        end
    end

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

outfile = @projectroot("results", "sim-cold-start.jld2");

@save outfile results
