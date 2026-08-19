using Intercepts
using Random
using DrWatson
using ProjectRoot
using JLD2

Random.seed!(1234)

# Imbalanced K=5 multinomial logistic. A rare free class gives a low-Rayleigh
# coordinate direction, while the rare reference class gives a low-Rayleigh
# common-shift direction. The gradient strategy attenuates both through
# H_00/L_0.
class_probs = [0.7, 0.1, 0.1, 0.05, 0.05]

param_dict = Dict{String, Any}(
    "it" => collect(1:3),
    "n" => [500],
    "p" => [200],
    "s" => [5],
    "reg" => [0.05],
    "K" => [5],
    "strategy" => [:gradient, :newton, :exact],
)

params = dict_list(param_dict)

results = []

for (i, d) in enumerate(params)
    @unpack it, n, p, s, reg, K, strategy = d

    Random.seed!(it)

    res = simulated_experiment(
        n,
        p;
        response = :multinomial,
        strategy = strategy,
        K = K,
        class_probs = class_probs,
        s = s,
        reg = reg,
        ρ = 0.3,
        amplitude = 0.5,
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

# Score each run as relative primal suboptimality against a shared optimum F*
# (best primal across strategies for the seed), with the duality gap certifying
# F*. Per seed, since each seed is a distinct random problem.
suboptimality_against_certified_optimum!(results; instance_of = r -> r["it"])

outfile = @projectroot("results", "sim-multinomial-imbalance.jld2");

@save outfile results
