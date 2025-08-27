using Intercepts
using Random
using GLM
using Statistics
using CairoMakie
using DrWatson
using ProjectRoot
using JLD2

Random.seed!(1234);

function simulated_experiment(
    n = 100,
    p = 10000;
    response = :binomial,
    strategy = :gradient,
    μ0 = 0.9,
    s = 10,
    reg = 0.01,
    randomize = false,
    update_freq = 1,
    maxit = 1000,
    maxtime = Inf,
    ρ = 0.6,
    amplitude = 1.0,
    x_density = 1.0,
    x_type = :normal,
    means = :random,
)
    X, y = generatedata(
        n,
        p;
        response = response,
        μ0 = μ0,
        x_type = x_type,
        x_density = x_density,
        ρ = ρ,
        s = s,
        amplitude = amplitude,
        means = means,
    )

    lossfun = Quadratic()

    if response == :binomial
        lossfun = Logistic()
    else
        error("Unknown response type: $response")
    end

    if strategy == :gradient
        intercept_strategy = GradientStrategy()
    elseif strategy == :newton
        intercept_strategy = NewtonStrategy()
    elseif strategy == :exact
        intercept_strategy = ExactStrategy()
    else
        error("Unknown strategy: $strategy")
    end

    res = cdsolver(
        X,
        y,
        reg,
        lossfun = lossfun,
        intercept_strategy = intercept_strategy,
        maxit = maxit,
        maxtime = maxtime,
        randomize = randomize,
        update_freq = update_freq,
    )

    return (
        time = res.time,
        gaps = res.gaps,
        relgaps = res.relgaps,
        primals = res.primals,
        duals = res.duals,
    )
end;

param_dict = Dict{String,Any}(
    "it" => collect(1:1),
    "n" => [100],
    "p" => [1000],
    "s" => [10],
    "reg" => [0.02],
    "μ0" => [0.5],
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

outfile = @projectroot("results", "sim-logreg-cd.jld2");

@save outfile results
