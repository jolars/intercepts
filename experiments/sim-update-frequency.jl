# Compare intercept-update frequencies in logistic coordinate descent.
# Use the Newton strategy for all synthetic and real-data fits.
# Run from the project root after staging the real-data archive:
#   julia --project=. experiments/sim-update-frequency.jl

using Intercepts, Random, LinearAlgebra, DataFrames, CSV, ProjectRoot

const N = 500
const P = 1000
const SEEDS = 1:5
const MU0 = [0.5, 0.9, 0.99]
const REG = 0.05
const TOL = 1.0e-8
const MAXIT = 20_000
const REPS = 3
const DATASETS = [
    ("w1a", (;)),
    ("a4a", (;)),
    ("gisette_scale", (;)),
    (
        "news20-imb", (
            dataset = "news20.binary", n_positive = 60,
            n_negative = 1940, min_nnz_per_column = 20, seed = 42,
        ),
    ),
]

# Seven logarithmically spaced coordinate strides, including both endpoints.
stride_ladder(p) = round.(Int, exp.(range(0, log(p); length = 7)))

solve(X, y, k; maxit = MAXIT) = cdsolver(
    X, y, REG;
    lossfun = LogisticLoss(), intercept_strategy = NewtonStrategy(), k = k,
    tol = TOL, maxit = maxit, randomize = false,
)

function run_problem!(rows, dataset, X, y; mu = missing, seed = missing, reps = 1)
    n, p = size(X)
    ladder = stride_ladder(p)
    # Warm up the solver's dense or sparse specialization before timing.
    solve(X, y, 1; maxit = 3)
    solve(X, y, p; maxit = 3)
    for rep in 1:reps, k in ladder
        GC.gc()
        res = solve(X, y, k)
        row = (
            dataset = dataset, μ0 = mu, n = n, p = p, seed = seed, rep = rep,
            k = k, updates_per_pass = cld(p, k), passes = res.passes,
            wall_time = res.time[end], final_relgap = res.relgaps[end],
            converged = res.relgaps[end] <= TOL,
        )
        push!(rows, row)
        println(
            "[$(length(rows))/189] $dataset μ0=$mu seed=$seed rep=$rep k=$k: ",
            "$(row.passes) passes, $(round(row.wall_time; digits = 3))s, converged=$(row.converged)"
        )
        flush(stdout)
    end
    return nothing
end

function main()
    BLAS.set_num_threads(1)
    rows = NamedTuple[]
    for mu in MU0, seed in SEEDS
        Random.seed!(seed)
        X, y = generatedata(
            N, P; response = :binomial, μ0 = mu, x_type = :normal,
            ρ = 0.6, s = 10, amplitude = 1.0,
        )
        run_problem!(rows, "simulated", X, Float64.(y); mu = mu, seed = seed)
    end
    for (name, kwargs) in DATASETS
        dataset = get(kwargs, :dataset, name)
        loader = (key => value for (key, value) in pairs(kwargs) if key != :dataset)
        X, y = prepare_real_data(dataset; loader...)
        run_problem!(rows, name, X, y; reps = REPS)
    end
    df = DataFrame(rows)
    sort!(df, [:dataset, :μ0, :seed, :rep, :k])
    @assert nrow(df) == 189
    @assert nrow(unique(select(df, [:dataset, :μ0, :seed, :rep, :k]))) == nrow(df)
    @assert all(df.passes .> 0)
    @assert all(isfinite, df.wall_time) && all(df.wall_time .> 0)
    @assert all(isfinite, df.final_relgap)
    all(df.converged) || error("Some fits did not reach the relative duality-gap tolerance")
    outfile = @projectroot("results", "sim-update-frequency.csv")
    CSV.write(outfile, df)
    return println("Saved $(nrow(df)) converged fits to $outfile.")
end

main()
