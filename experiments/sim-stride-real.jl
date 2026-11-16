# Real-data check on the intercept stride (reviewer comment 3).
#
# sim-stride-logistic.jl found that firing the 1/L0 intercept every k < p
# coordinates recovers the whole gap to the Armijo-guarded Newton intercept,
# and that the size of that gap tracks the curvature ratio
# H00/L0 = 4 * mean(pi_i (1 - pi_i)), which collapses as the response becomes
# imbalanced. That is a prediction, so this script tests it out of sample on
# the shipped LIBSVM datasets rather than re-measuring the same surface:
#
#   w1a           pi+ = 0.029   severely imbalanced  -> predict a large payoff
#   news20-imb    pi+ = 0.030   imbalanced, wide     -> predict a large payoff
#   a4a           pi+ = 0.249   moderate             -> baseline only, see below
#   gisette_scale pi+ = 0.500   balanced             -> predict no payoff
#
# gisette is the control: if striding speeds it up, the mechanism is not the
# one claimed. a4a runs at the baseline only because `k` must divide `p` and
# p = 123 = 3 * 41 admits just {1, 3, 41} updates per pass, which cannot
# resolve the optimum the other datasets place near p/k = 10. It still
# contributes a handicap measurement at the shared baseline.
#
# The solver is deterministic here (randomize = false, fixed data), so pass
# counts repeat exactly across reps and only wall time carries noise; REPS
# exists for the timings, not for the pass counts. Each dataset is checkpointed
# under results/sim-stride-real/cells/ so an interrupted run resumes instead of
# starting over; the assembled JLD2 and CSV are the tracked artifacts.

using DrWatson
@quickactivate "Intercepts"
using Intercepts, DataFrames, JLD2, Statistics, Printf, ProjectRoot, CSV

const REG = 0.05
const MAXIT = 20_000
const MAXTIME = 300.0
const TOL = 1.0e-10
const REPS = 5
const STRATEGIES = [:gradient, :newton]
const TARGET_RATIOS = [1, 2, 5, 10, 25, 50]

const CELLDIR = @projectroot("results", "sim-stride-real", "cells")
mkpath(CELLDIR)

# name => (loader kwargs..., sweep). "news20-imb" is the subsampled
# news20.binary already used by sim-real-logreg.jl: 2 000 rows at 3 % positive,
# with features appearing in fewer than 20 of them dropped so standardisation
# stays stable. `sweep = false` runs the k = p baseline only.
const DATASETS = [
    ("w1a", (sweep = true,)),
    ("a4a", (sweep = false,)),
    ("gisette_scale", (sweep = true,)),
    (
        "news20-imb",
        (
            sweep = true, dataset = "news20.binary", n_positive = 60,
            n_negative = 1940, min_nnz_per_column = 20, seed = 42,
        ),
    ),
]

divisors(p) = sort(unique(vcat([[d, p ÷ d] for d in 1:isqrt(p) if p % d == 0]...)))

"""
Strides to sweep for a dataset with `p` features.

`k` must divide `p`, so the targets in `TARGET_RATIOS` are generally not
attainable exactly; take the divisor closest to `p / r` for each target and
report the ratio actually achieved. `k = p` (one update per pass, at its end)
is the baseline and is always present.
"""
function stride_ladder(p)
    divs = divisors(p)
    return sort(unique([divs[argmin(abs.(divs .- p / r))] for r in TARGET_RATIOS]); rev = true)
end

"Curvature ratio H00/L0 at the fitted model; L0 = 1/4 for the logistic loss."
function curvature_ratio(X, y, res)
    η = X * res.coef .+ res.intercept
    w = hessian(LogisticLoss(), η, y)
    return sum(w) / (0.25 * length(y))
end

solve(X, y, strategy, k) = cdsolver(
    X, y, REG;
    lossfun = LogisticLoss(),
    intercept_strategy = Intercepts.get_intercept_strategy(strategy),
    maxit = MAXIT, maxtime = MAXTIME, tol = TOL, randomize = false, k = k,
)

"Run every strategy/stride/rep for one dataset and return its rows."
function run_dataset(name, kw)
    ds = get(kw, :dataset, name)
    loader = (key => v for (key, v) in pairs(kw) if key ∉ (:dataset, :sweep))
    X, y = prepare_real_data(ds; loader...)
    n, p = size(X)
    ladder = kw.sweep ? stride_ladder(p) : [p]
    π₊ = sum(y) / n
    @info "dataset" name n p π₊ ladder
    flush(stdout)

    rows = NamedTuple[]
    for strategy in STRATEGIES, k in ladder, rep in 1:REPS
        GC.gc()
        res = solve(X, y, strategy, k)
        push!(
            rows,
            (
                dataset = name, n = n, p = p, pi_plus = π₊,
                strategy = String(strategy), k = k, ratio = p / k, rep = rep,
                passes = res.passes, wall_time = res.time[end],
                h00_l0 = curvature_ratio(X, y, res),
                final_relgap = res.relgaps[end],
                converged = res.relgaps[end] <= TOL,
            ),
        )
        rep == REPS && @printf(
            "  %-14s %-9s k=%-6d p/k=%6.1f  passes=%6d  t=%7.2fs  conv=%s\n",
            name, strategy, k, p / k, res.passes, res.time[end],
            res.relgaps[end] <= TOL
        )
        rep == REPS && flush(stdout)
    end
    return rows
end

# Warm up both strategies so JIT does not land on a timed run.
let (Xw, yw) = prepare_real_data("breast-cancer")
    for s in STRATEGIES
        solve(Xw, yw, s, size(Xw, 2))
    end
end
GC.gc()

all_rows = NamedTuple[]
t_start = time()

for (name, kw) in DATASETS
    cell = joinpath(CELLDIR, "$(name).jld2")
    if isfile(cell)
        @info "resuming from checkpoint" name cell
        append!(all_rows, load(cell, "rows"))
        continue
    end
    rows = run_dataset(name, kw)
    jldsave(cell; rows = rows)
    append!(all_rows, rows)
    @printf("  -> checkpointed %s after %.0fs\n", name, time() - t_start)
    flush(stdout)
end

df = DataFrame(all_rows)
n_censored = count(!, df.converged)
n_censored > 0 && @warn "censored runs present" n_censored total = nrow(df)

@save @projectroot("results", "sim-stride-real.jld2") df
CSV.write(@projectroot("results", "sim-stride-real.csv"), df)
@printf("\nSaved %d rows; %d censored.\n", nrow(df), n_censored)
