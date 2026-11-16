using Intercepts
using Random
using DrWatson
using ProjectRoot
using JLD2
using DataFrames
using CSV
using Statistics

# Logistic intercept-stride sweep.
#
# Question 1 (parameterization). Is the best stride constant in p/k (a fraction
# of the features) or constant in k (an absolute number of coordinates)? The
# ladder below specifies target p/k ratios, rounding k up to an integer and
# reporting the achieved p/k. The chosen feature counts hit every target
# exactly while p varies 8x. If the optimum is relative, the argmin lines up
# across rows; if absolute, it drifts by 8x.
#
# Rows (n, p) are chosen so that n/p and p are not fully confounded:
#   (1000,  250)  n/p = 4     reference
#   (1000, 1000)  n/p = 1     4x the p of row 1, same n
#   (1000, 2000)  n/p = 0.5   8x the p of row 1, same n
#   (4000, 1000)  n/p = 4     same n/p as row 1, but 4x the p
# Row 4 against row 1 separates "same n/p" from "same p"; row 4 against row 2
# separates "same p" from "same n".
#
# Question 2 (theory). @eq-grad-residual says the 1/L0 step removes only a
# fraction H00/L0 of the coupling. For logistic loss H00 = sum_i p_i(1 - p_i)
# and the gradient step divides by L0 * n = n/4, so the curvature ratio is
# 4 * mean(p_i (1 - p_i)), recorded here at the fitted solution. Plotting the
# measured gradient speedup against it tests the prediction rather than only
# mapping the surface.

const SEEDS = collect(1:10)
const STRIDE_RATIOS = [1, 2, 5, 10, 25, 50, 125, 250]   # p / k
const CONFIGS = [(n = 1000, p = 250), (n = 1000, p = 1000),
    (n = 1000, p = 2000), (n = 4000, p = 1000)]
const MU0 = [0.5, 0.9, 0.99]
const STRATEGIES = [:gradient, :newton]

const S_ACTIVE = 10
const REG = 0.05
const RHO = 0.6
const AMPLITUDE = 1.0
const MAXIT = 20_000
const MAXTIME = 120.0
const TOL = 1.0e-10

strategy_object(s) = s === :gradient ? GradientStrategy() :
    s === :newton ? NewtonStrategy() : error("Unknown strategy: $s")

"""
Solve one cell and return the run plus diagnostics evaluated at the fit.
`k` is the coordinate stride; the data are regenerated from `seed` inside the
call so every cell with the same seed solves the same problem.
"""
function solve_cell(n, p, μ0, strategy, k, seed; maxit = MAXIT)
    Random.seed!(seed)
    X, y = generatedata(
        n, p; response = :binomial, μ0 = μ0,
        x_type = :normal, ρ = RHO, s = S_ACTIVE, amplitude = AMPLITUDE,
    )
    y = Float64.(y)

    res = cdsolver(
        X, y, REG;
        lossfun = LogisticLoss(), intercept_strategy = strategy_object(strategy),
        k = k, randomize = false, tol = TOL, maxit = maxit, maxtime = MAXTIME,
    )

    # Curvature ratio at the fitted solution: H00 / (L0 * n) = 4 * mean(p(1-p)).
    η = X * res.coef .+ res.intercept
    w = hessian(LogisticLoss(), η, y)
    return res, sum(w) / (0.25 * n), mean(y)
end

params = [
    (n = cfg.n, p = cfg.p, μ0 = μ0, strategy = strat, ratio = cfg.p / k, k = k, seed = seed)
        for cfg in CONFIGS for μ0 in MU0 for strat in STRATEGIES
        for k in sort(unique(cld.(cfg.p, STRIDE_RATIOS)); rev = true) for seed in SEEDS
]

# Warm up each strategy so JIT does not land on the timed runs.
for strat in STRATEGIES
    solve_cell(50, 20, 0.5, strat, 20, 1; maxit = 3)
end
GC.gc()

rows = Vector{NamedTuple}()
t_start = time()

for (i, d) in enumerate(params)
    GC.gc()
    res, h00_l0, pi_plus = solve_cell(d.n, d.p, d.μ0, d.strategy, d.k, d.seed)
    converged = res.relgaps[end] <= TOL

    push!(
        rows,
        (
            n = d.n, p = d.p, np_ratio = d.n / d.p, μ0 = d.μ0,
            strategy = String(d.strategy), ratio = d.ratio, k = d.k, seed = d.seed,
            updates_per_pass = cld(d.p, d.k),
            passes = res.passes, wall_time = res.time[end],
            h00_l0 = h00_l0, pi_plus = pi_plus,
            final_relgap = res.relgaps[end], converged = converged,
        ),
    )

    if i % 25 == 0 || i == length(params)
        el = round(time() - t_start; digits = 1)
        println(
            "[$i/$(length(params))] n=$(d.n) p=$(d.p) μ0=$(d.μ0) $(d.strategy) " *
                "p/k=$(d.ratio) seed=$(d.seed) passes=$(res.passes) " *
                "t=$(round(res.time[end], digits = 2))s conv=$converged  [$(el)s elapsed]",
        )
        flush(stdout)
    end
end

df = DataFrame(rows)
n_censored = count(!, df.converged)
n_censored > 0 && @warn "cells hit maxit/maxtime without reaching tol" n_censored

@save @projectroot("results", "sim-stride-logistic.jld2") df
CSV.write(@projectroot("results", "sim-stride-logistic.csv"), df)
println("\nSaved $(nrow(df)) rows; $n_censored censored.")
