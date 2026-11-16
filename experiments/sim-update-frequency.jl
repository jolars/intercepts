using Intercepts
using Random
using DrWatson
using ProjectRoot
using JLD2

# Intercept update stride k, across all four response families.
#
# The drift calculation (@eq-intercept-drift) says each coefficient update moves
# the intercept's conditional optimum, so an intercept solved at the end of a
# pass is already stale by the next coordinate. This sweep asks whether firing
# the intercept update every k coordinates recovers that staleness, and whether
# frequent cheap 1/L0 steps substitute for one Newton step per pass.
#
# k is the coordinate stride: a pass performs cld(p, k) updates, each costing
# O(n) against an O(np) pass, so the overhead is roughly 1/k of a pass. k = p
# is the paper's baseline (one update per pass, at its end); k = 1 updates
# after every coordinate. A shorter final block still updates the intercept.
#
# What each family contributes:
#   normal       L0 = H00 = 1 exactly, so the 1/L0 step IS the Newton step and
#                one Newton step solves the quadratic intercept subproblem
#                exactly. All strategies coincide; any k-dependence that
#                survives is pure drift, with the curvature deficit removed.
#   binomial     H00/L0 = 4*pi*(1-pi) falls with imbalance: the live case.
#   poisson      L0 = Inf, so the gradient step is identically zero at every k
#                and there is nothing to sweep. Excluded below; the live
#                comparison is Newton against exact.
#   multinomial  vector intercept against the scalar Bohning bound L0 = 1/2.

Random.seed!(1234)

const MAXIT = 20_000
const MAXTIME = 300.0
const TOL = 1.0e-10

# Each cell is solved on SEEDS independent problem instances. Reported passes
# and wall time are medians over seeds, so the spread across seeds is the
# error bar rather than a repeated timing of one fixed problem.
#
# Timing hygiene: the per-pass wall-clock trajectory is a single time() sample
# per pass, so JIT compilation and GC pauses land directly on the reported
# axis. Each strategy is warmed up once to compile its solver specialization
# and the heap is collected before every timed run.
const SEEDS = collect(1:5)

# Include the once-per-pass baseline even when p is outside the fixed ladder.
const K_LADDER = [1, 5, 10, 50, 100, 250, 500, 1000]
k_ladder(p) = sort(unique([filter(k -> k <= p, K_LADDER); p]))

const MULTINOMIAL_CLASS_PROBS = Dict(
    :balanced => [0.2, 0.2, 0.2, 0.2, 0.2],
    :imbalanced => [0.7, 0.1, 0.1, 0.05, 0.05],
)

# Matches the paper's harness: the multinomial exact strategy uses the looser
# gradient tolerance that `run_solver` applies there.
strategy_object(s, family) = s === :gradient ? GradientStrategy() :
    s === :newton ? NewtonStrategy() :
    s === :exact ?
    (family == "multinomial" ? ExactStrategy(gradient_tol = 1.0e-8) : ExactStrategy()) :
    error("Unknown strategy: $s")

# Per-family grids. Each entry fixes the problem; k and strategy sweep within.
grids = [
    Dict{String, Any}(
        "family" => "normal", "n" => [500], "p" => [1000], "s" => [10],
        "reg" => [0.05], "μ0" => [0.5], "regime" => ["identity-link"],
        "strategy" => [:gradient, :newton], "k" => k_ladder(1000), "seed" => SEEDS,
    ),
    Dict{String, Any}(
        "family" => "binomial", "n" => [500], "p" => [1000], "s" => [10],
        "reg" => [0.05], "μ0" => [0.5, 0.9, 0.99], "regime" => ["prevalence"],
        "strategy" => [:gradient, :newton, :exact], "k" => k_ladder(1000), "seed" => SEEDS,
    ),
    Dict{String, Any}(
        "family" => "poisson", "n" => [500], "p" => [1000], "s" => [10],
        "reg" => [0.1], "μ0" => [10.0, 100.0], "regime" => ["mean-rate"],
        "strategy" => [:newton, :exact], "k" => k_ladder(1000), "seed" => SEEDS,
    ),
    Dict{String, Any}(
        "family" => "multinomial", "n" => [500], "p" => [200], "s" => [5],
        "reg" => [0.05], "μ0" => [0.0], "regime" => ["balanced", "imbalanced"],
        "strategy" => [:gradient, :newton, :exact], "k" => k_ladder(200), "seed" => SEEDS,
    ),
]

"""
Solve one cell. The data are regenerated from a fixed seed inside the call so
every repetition solves exactly the same problem.
"""
function solve_cell(family, n, p, s, reg, μ0, regime, strategy, k, seed; maxit = MAXIT)
    Random.seed!(seed)
    intercept_strategy = strategy_object(strategy, family)

    if family == "multinomial"
        X, y = generatedata(
            n, p; response = :multinomial, K = 5,
            class_probs = MULTINOMIAL_CLASS_PROBS[Symbol(regime)],
            x_type = :normal, ρ = 0.3, s = s, amplitude = 0.5, means = :random,
        )
        return multinomial_cdsolver(
            X, y, reg;
            lossfun = MultinomialLogisticLoss(K = 5, lipschitz = 0.5),
            intercept_strategy = intercept_strategy, k = k,
            randomize = false, tol = TOL, maxit = maxit, maxtime = MAXTIME,
        )
    end

    response = Symbol(family)
    X, y = generatedata(
        n, p; response = response, μ0 = μ0,
        x_type = :normal, ρ = 0.6, s = s, amplitude = 1.0,
    )
    lossfun = response === :normal ? QuadraticLoss() :
        response === :binomial ? LogisticLoss() : PoissonLoss()
    return cdsolver(
        X, Float64.(y), reg;
        lossfun = lossfun, intercept_strategy = intercept_strategy, k = k,
        randomize = false, tol = TOL, maxit = maxit, maxtime = MAXTIME,
    )
end

params = reduce(vcat, dict_list.(grids))

# Warm up every (family, strategy) pair on a tiny problem so the timed runs do
# not pay for JIT specialization of the solver.
let
    warmup = unique((d["family"], d["strategy"]) for d in params)
    for (family, strategy) in warmup
        n_w, p_w = 50, 20
        solve_cell(
            family, n_w, p_w, 2, 0.1,
            family == "poisson" ? 10.0 : 0.5,
            family == "multinomial" ? "balanced" : "warmup",
            strategy, p_w, 1; maxit = 3,
        )
    end
    GC.gc()
end

results = Vector{Dict{String, Any}}()

for (i, d) in enumerate(params)
    @unpack family, n, p, s, reg, μ0, regime, strategy, k, seed = d

    GC.gc()
    res = solve_cell(family, n, p, s, reg, μ0, regime, strategy, k, seed)

    T = length(res.primals)
    inner_steps = family == "multinomial" ? Int[] : res.inner_steps

    d_exp = Dict{String, Any}(
        "family" => family, "n" => n, "p" => p, "s" => s, "reg" => reg,
        "μ0" => μ0, "regime" => regime, "strategy" => String(strategy),
        "k" => k, "updates_per_pass" => cld(p, k), "seed" => seed,
        "primals" => res.primals, "duals" => res.duals,
        "gaps" => res.gaps, "relgaps" => res.relgaps,
        "pass" => collect(1:T), "time" => res.time[1:T],
        "wall_time" => res.time[end],
        "inner_steps" => inner_steps,
        "cum_inner_steps" => isempty(inner_steps) ? Int[] : cumsum(inner_steps),
        "passes" => res.passes,
    )

    println(
        "[$i/$(length(params))] $family/$regime μ0=$μ0 $strategy k=$k " *
            "(updates/pass=$(cld(p, k))) passes=$(res.passes) " *
            "t=$(round(res.time[end], digits = 2))s relgap=$(res.relgaps[end])",
    )

    push!(results, d_exp)
end

suboptimality_against_shared_dual!(
    results;
    instance_of = r -> (
        r["family"], r["regime"], r["n"], r["p"], r["s"], r["reg"], r["μ0"], r["seed"],
    ),
)

# The per-pass histories above exist so the shared dual certificate can score
# every run of an instance against the same bound. Once that has run they are
# dead weight: nothing downstream reads them, and keeping them makes the cached
# artifact ~20 MB instead of ~200 KB. Drop them and keep the final values.
const HISTORY_KEYS = [
    "primals", "duals", "gaps", "relgaps", "pass", "time",
    "inner_steps", "cum_inner_steps", "primal_relgaps", "dual_relgaps",
]

for r in results
    r["final_relgap"] = last(r["relgaps"])
    r["final_gap"] = last(r["gaps"])
    r["final_primal_relgap"] = last(r["primal_relgaps"])
    r["final_dual_relgap"] = last(r["dual_relgaps"])
    r["total_inner_steps"] = isempty(r["inner_steps"]) ? 0 : last(r["cum_inner_steps"])
    foreach(key -> delete!(r, key), HISTORY_KEYS)
end

outfile = @projectroot("results", "sim-update-frequency.jld2")
@save outfile results
println("Saved results to $outfile")
