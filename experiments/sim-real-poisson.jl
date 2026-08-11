using Intercepts
using Random
using DrWatson
using ProjectRoot
using DataFrames
using JLD2
using LinearAlgebra
using CSV
using JSON3

Random.seed!(42);

# Poisson real-data analogue of sim-real-logreg.jl. Loads the pre-standardised
# congress109 problem from results/real-solvers-poisson/congress109/ (produced
# by experiments/sim-real-poisson-problem.R), runs cdsolver with PoissonLoss
# under four intercept strategies, and saves convergence trajectories to
# results/real-poisson.jld2 for the paper's fig-real-poisson chunk.
#
# Includes :unguarded_newton so the real-data panel can show the same
# Armijo-load-bearing story as the synthetic sim-cold-start-poisson.jl
# experiment: the unguarded Newton intercept update at cold start lands near
# ybar - 1 ~ 10.8 while the exact intercept is log(ybar) ~ 2.5, a ~50000x
# loss-value overshoot that the guarded variant absorbs monotonically and
# the unguarded variant must recover from.

probdir = @projectroot("results", "real-solvers-poisson", "congress109")
X = Matrix(CSV.read(joinpath(probdir, "X.csv"), DataFrame; header = false))
ydf = CSV.read(joinpath(probdir, "y.csv"), DataFrame)
y = Float64.(ydf.y)
meta = JSON3.read(read(joinpath(probdir, "meta.json"), String))
reg = meta["reg"]

n, p = size(X)
@info "Loaded congress109 problem" n p target_phrase = meta["target_phrase"] ybar = meta[
    "y_mean",
] reg

strategy_map = Dict(
    :gradient => GradientStrategy(),
    :newton => NewtonStrategy(),
    :exact => ExactStrategy(),
    :unguarded_newton => UnguardedNewtonStrategy(),
)

param_dict = Dict{String, Any}("strategy" => [
    :gradient,
    :newton,
    :exact,
    :unguarded_newton,
]);

params = dict_list(param_dict);

results = [];

for (i, d) in enumerate(params)
    @unpack strategy = d
    @info "Running" strategy

    res = try
        cdsolver(X, y, reg;
            lossfun = PoissonLoss(),
            intercept_strategy = strategy_map[strategy],
            randomize = false,
            normalization = :none,  # X is already standardised on disk
            maxit = 1000,
            maxtime = 120.0,
        )
    catch err
        @warn "solver failed" strategy err
        (
            time = Float64[],
            gaps = Float64[],
            relgaps = Float64[],
            primals = Float64[],
            duals = Float64[],
        )
    end

    d_exp = Dict{String, Any}(copy(d))
    d_exp["time"] = res.time
    d_exp["gaps"] = res.gaps
    d_exp["relgaps"] = res.relgaps
    d_exp["primals"] = res.primals
    d_exp["dataset"] = "congress109"

    push!(results, d_exp)
end

outfile = @projectroot("results", "real-poisson.jld2");

@save outfile results
