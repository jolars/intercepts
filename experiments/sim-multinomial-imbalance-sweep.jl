using Intercepts
using Random
using DrWatson
using ProjectRoot
using JLD2

Random.seed!(1234)

# Sweep of class imbalance (p_K, the rarest-class marginal) × feature amplitude
# for the K=5 multinomial logistic problem of sim-multinomial-imbalance.jl,
# testing whether the collapse of {bare Newton, damped Newton, convergence}
# observed at the single anchor point persists across the steelman range
# (down to p_K = 0.005 and up to four-times the figure amplitude).
#
# class_probs is built so the anchor cell (p_K = 0.05) reproduces the
# single-config experiment exactly: [0.7 + (0.05 - p_K), 0.1, 0.1, 0.05, p_K].

const P_K_LEVELS = [0.05, 0.02, 0.01, 0.005]
const AMP_LEVELS = [0.5, 1.0, 1.5, 2.0]

function build_class_probs(p_K::Real)
    dominant = 0.7 + (0.05 - p_K)
    return [dominant, 0.1, 0.1, 0.05, p_K]
end

param_dict = Dict{String,Any}(
    "it" => collect(1:3),
    "n" => [500],
    "p" => [200],
    "s" => [5],
    "reg" => [0.05],
    "K" => [5],
    "p_K" => P_K_LEVELS,
    "amplitude" => AMP_LEVELS,
    "strategy" => [:gradient, :newton, :damped_newton, :exact],
)

params = dict_list(param_dict)

const MAXIT = 1000
const TOL = 1.0e-6

results = Vector{Dict{String,Any}}()

for (i, d) in enumerate(params)
    @unpack it, n, p, s, reg, K, p_K, amplitude, strategy = d

    Random.seed!(it)

    class_probs = build_class_probs(p_K)

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
        amplitude = amplitude,
        randomize = false,
        maxit = MAXIT,
    )

    relgaps = res.relgaps
    idx_tol = findfirst(g -> g <= TOL, relgaps)
    reached = idx_tol !== nothing
    passes = reached ? idx_tol : length(relgaps)

    d_exp = Dict{String,Any}(copy(d))
    d_exp["strategy"] = String(strategy)
    d_exp["time"] = res.time
    d_exp["gaps"] = res.gaps
    d_exp["relgaps"] = res.relgaps
    d_exp["primals"] = res.primals
    d_exp["pass"] = collect(1:length(res.relgaps))
    d_exp["passes"] = passes
    d_exp["reached_tol"] = reached
    d_exp["final_relgap"] = max(relgaps[end], 1.0e-20)

    println(
        "[$i/$(length(params))] p_K=$p_K amp=$amplitude strategy=$strategy seed=$it " *
        "passes=$passes reached=$reached final_relgap=$(d_exp["final_relgap"])",
    )

    push!(results, d_exp)
end

outfile = @projectroot("results", "sim-multinomial-imbalance-sweep.jld2")
@save outfile results
println("Saved results to $outfile")
