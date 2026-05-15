using Intercepts
using Random
using LinearAlgebra
using SparseArrays
using LIBSVMdata
using DrWatson
using ProjectRoot
using JLD2
using Printf

const K_GRID = 20
const EPSILON = 5.0e-3
const TOL = 1.0e-8
const MAXIT = 300
const MAXTIME_PER_SUBPROBLEM = 20.0

function load_problem(name::Symbol)
    if name == :simulated
        Random.seed!(2025)
        X, y = generatedata(
            500,
            1000;
            response = :binomial,
            μ0 = 0.9,
            x_type = :normal,
            ρ = 0.6,
            s = 10,
            amplitude = 1.0,
        )
        return X, Float64.(y)
    elseif name == :w1a
        X, y = load_dataset("w1a", verbose = false)
        y = Float64.(Int.(y .== 1))
        return X, y
    else
        error("Unknown dataset: $name")
    end
end

function run_path(X, y, strategy_sym, randomize, warm_start)
    n = length(y)
    lossfun = LogisticLoss()
    strategy = if strategy_sym == :gradient
        GradientStrategy()
    elseif strategy_sym == :newton
        NewtonStrategy()
    elseif strategy_sym == :exact
        ExactStrategy()
    else
        error("Unknown strategy: $strategy_sym")
    end

    λmax = lambdamax(lossfun, X, y)
    reg_grid = geomspace(1.0, EPSILON, K_GRID)  # ε·λmax to λmax in λ; here as reg = λ/λmax

    passes = zeros(Int, K_GRID)
    times = zeros(Float64, K_GRID)
    init_relgap = zeros(Float64, K_GRID)
    final_relgap = zeros(Float64, K_GRID)
    init_partial0F = zeros(Float64, K_GRID)
    nnz_solution = zeros(Int, K_GRID)

    coef_carry = nothing
    intercept_carry = 0.0

    for (k, reg) in enumerate(reg_grid)
        if warm_start && coef_carry !== nothing
            ci, ii = coef_carry, intercept_carry
        else
            ci, ii = nothing, 0.0
        end

        η_init = if ci === nothing
            zeros(n)
        else
            X * ci .+ ii
        end
        init_partial0F[k] = abs(sum(gradient(lossfun, η_init, y)))

        t_start = time()
        res = cdsolver(
            X,
            y,
            reg;
            lossfun = lossfun,
            intercept_strategy = strategy,
            randomize = randomize,
            tol = TOL,
            maxit = MAXIT,
            maxtime = MAXTIME_PER_SUBPROBLEM,
            coef_init = ci,
            intercept_init = ii,
        )
        elapsed = time() - t_start

        passes[k] = res.passes
        times[k] = isempty(res.time) ? 0.0 : res.time[end]
        init_relgap[k] = isempty(res.relgaps) ? NaN : res.relgaps[1]
        final_relgap[k] = isempty(res.relgaps) ? NaN : res.relgaps[end]
        nnz_solution[k] = count(!iszero, res.coef)

        coef_carry = res.coef
        intercept_carry = res.intercept

        @printf(
            "    λ_k=%d/%d reg=%.4g passes=%d wall=%.2fs final_relgap=%.2e\n",
            k,
            K_GRID,
            reg,
            res.passes,
            elapsed,
            final_relgap[k],
        )
        flush(stdout)
    end

    return (
        reg_grid = collect(reg_grid),
        passes = passes,
        times = times,
        init_relgap = init_relgap,
        final_relgap = final_relgap,
        init_partial0F = init_partial0F,
        nnz_solution = nnz_solution,
        λmax = λmax,
    )
end

param_dict = Dict{String,Any}(
    "dataset" => [:simulated, :w1a],
    "strategy" => [:gradient, :newton, :exact],
    "randomize" => [false, true],
    "warm_start" => [false, true],
)

params = dict_list(param_dict)
results = Vector{Dict{String,Any}}()

data_cache = Dict{Symbol,Tuple}()

for (i, d) in enumerate(params)
    @unpack dataset, strategy, randomize, warm_start = d

    X, y = get!(data_cache, dataset) do
        load_problem(dataset)
    end

    println(
        "[$(i)/$(length(params))] dataset=$dataset strategy=$strategy randomize=$randomize warm=$warm_start",
    )
    flush(stdout)

    path = run_path(X, y, strategy, randomize, warm_start)

    d_exp = Dict{String,Any}(
        "dataset" => String(dataset),
        "strategy" => String(strategy),
        "randomize" => randomize,
        "warm_start" => warm_start,
        "reg_grid" => path.reg_grid,
        "passes" => path.passes,
        "times" => path.times,
        "init_relgap" => path.init_relgap,
        "final_relgap" => path.final_relgap,
        "init_partial0F" => path.init_partial0F,
        "nnz_solution" => path.nnz_solution,
        "lambdamax" => path.λmax,
    )

    push!(results, d_exp)
end

outfile = @projectroot("results", "sim-warmstart-path.jld2")
@save outfile results
println("Saved results to $outfile")
