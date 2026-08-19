using Intercepts
using Random
using DrWatson
using ProjectRoot
using JLD2

Random.seed!(1234);

# Cold-start cyclic CD on Poisson regression at extreme rates.
# Poisson with log link has f''(η) = e^η and f'''(η) = e^η, both unbounded as
# η grows. At cold start (β₀ = 0) on a regime with mean rate μ₀, the unguarded
# Newton intercept step lands near μ̄_y - 1 ≈ μ₀ - 1; the exact intercept is
# log(μ̄_y) ≈ log(μ₀). For μ₀ = 100 that is an ~22x overshoot, evaluating the
# loss at η ~ 99 where e^η ≈ 10^43. The Armijo guard inside NewtonStrategy
# should backtrack many times; the unguarded variant should diverge.
param_dict = Dict{String, Any}(
    "it" => collect(1:5),
    "n" => [500],
    "p" => [1000],
    "s" => [10],
    "reg" => [0.1],
    "μ0" => [10.0, 30.0, 100.0, 300.0],
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
        response = :poisson,
        strategy = strategy,
        μ0 = 10.0,
        s = 5,
        reg = 0.1,
        amplitude = 0.1,
        randomize = false,
        maxit = 3,
    )
end

nreps = 3;

finaltime(r) = isempty(r.time) ? Inf : r.time[end];

results = [];

for (i, d) in enumerate(params)
    @unpack it, n, p, s, reg, μ0, strategy = d

    res = nothing
    for rep in 1:nreps
        Random.seed!(it)
        GC.gc()

        r = try
            simulated_experiment(
                n,
                p;
                response = :poisson,
                strategy = strategy,
                μ0 = μ0,
                s = s,
                reg = reg,
                amplitude = 0.1,
                randomize = false,
                maxit = 1000,
            )
        catch err
            # Unguarded Newton can overflow to Inf/NaN at extreme μ₀; record an
            # empty trajectory rather than crashing the sweep.
            @warn "solver failed" strategy μ0 it err
            (
                time = Float64[],
                gaps = Float64[],
                relgaps = Float64[],
                primals = Float64[],
                duals = Float64[],
            )
        end

        if res === nothing || finaltime(r) < finaltime(res)
            res = r
        end
    end

    d_exp = Dict{String, Any}(copy(d))
    d_exp["time"] = res.time
    d_exp["gaps"] = res.gaps
    d_exp["relgaps"] = res.relgaps
    d_exp["primals"] = res.primals

    push!(results, d_exp)
end

outfile = @projectroot("results", "sim-cold-start-poisson.jld2");

@save outfile results
