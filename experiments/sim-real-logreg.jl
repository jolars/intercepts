using Intercepts
using CSV
using DataFrames
using JLD2
using LinearAlgebra
using ProjectRoot
using Random
using SparseArrays
using Statistics

# --resume reuses completed runs; --rerun=DATASET refreshes a selected dataset.
const DATASETS = [
    "w1a", "w8a", "news20-imb", "a4a", "leukemia", "breast-cancer", "gisette_scale",
]
const STRATEGIES = (:gradient, :newton, :exact)
const REG = 0.05
const REFERENCE_TOL = 1.0e-10
const CACHE = @projectroot("results", "real-logreg.jld2")
const DIAGNOSTICS = @projectroot("results", "real-logreg-diagnostics.csv")

display_name(dataset) = replace(dataset, "news20-imb" => "news20-3pct", "gisette_scale" => "gisette")

function prepare_problem(dataset)
    if dataset == "news20-imb"
        return prepare_real_data(
            "news20.binary";
            n_positive = 60, n_negative = 1940, min_nnz_per_column = 20, seed = 42,
        )
    end
    return prepare_real_data(dataset)
end

function fitted_diagnostics(dataset, X, y, result)
    n, p = size(X)
    eta = X * result.coef .+ result.intercept
    fitted = invlink(LogisticLoss(), eta)
    pi_plus = mean(y)
    H00 = mean(fitted .* invlink(LogisticLoss(), -eta))
    intercept_residual = abs(mean(fitted) - pi_plus)
    last(result.relgaps) <= REFERENCE_TOL || error("$dataset reference missed tolerance")
    intercept_residual <= 1.0e-6 || error("reference intercept is not stationary")
    return (
        dataset = display_name(dataset), n = n, p = p,
        density = nnz(sparse(X)) / (n * p), reg = REG,
        pi_plus = pi_plus, intercept_only_fraction = 4pi_plus * (1 - pi_plus),
        H00 = H00, H00_over_L0 = 4H00, intercept_residual = intercept_residual,
        lambda = result.λ / n, lambda_max = result.λmax / n,
        reference_primal = last(result.primals) / n,
        reference_dual = last(result.duals) / n,
        reference_relative_gap = last(result.relgaps),
        reference_passes = length(result.primals) - 1,
    )
end

function trajectory(dataset, strategy, result)
    return Dict{String, Any}(
        "dataset" => dataset, "strategy" => strategy, "reg" => REG,
        "time" => result.time, "primals" => result.primals,
        "duals" => result.duals, "gaps" => result.gaps, "relgaps" => result.relgaps,
    )
end

function check_reference(dataset, reference, group)
    reference.reg == REG || error("cached reference has a different regularization")
    reference.reference_relative_gap <= REFERENCE_TOL || error("uncertified reference")
    isempty(group) && return
    lower = maximum(maximum(r["duals"]) for r in group)
    upper = minimum(minimum(r["primals"]) for r in group)
    margin = 1.0e-8 * max(abs(upper), 1.0)
    lower - margin <= reference.n * reference.reference_primal <= upper + margin ||
        error("$dataset reference disagrees with its cached objective bounds")
    return nothing
end

function main()
    all(arg -> arg == "--resume" || startswith(arg, "--rerun="), ARGS) ||
        error("usage: sim-real-logreg.jl [--resume] [--rerun=DATASET]")
    resume = "--resume" in ARGS
    rerun = [split(arg, "="; limit = 2)[2] for arg in ARGS if startswith(arg, "--rerun=")]
    all(dataset -> dataset in DATASETS, rerun) || error("unknown dataset in --rerun")
    isempty(rerun) || resume || error("--rerun requires --resume")
    cached = resume && isfile(CACHE) ? JLD2.load(CACHE, "results") : []
    references = Dict{String, NamedTuple}()
    if resume && isfile(DIAGNOSTICS)
        for row in CSV.File(DIAGNOSTICS)
            references[row.dataset] = NamedTuple(row)
        end
    end

    Random.seed!(0)
    X_warm, y_warm = generatedata(30, 10; response = :binomial, s = 3)
    for strategy in (GradientStrategy(), NewtonStrategy(), ExactStrategy())
        cdsolver(
            sparse(X_warm), y_warm, REG;
            lossfun = LogisticLoss(), intercept_strategy = strategy, maxit = 3
        )
    end

    results = Any[]
    rows = NamedTuple[]
    for dataset in DATASETS
        group = dataset in rerun ? Any[] :
            Any[r for r in cached if r["dataset"] == dataset && r["reg"] == REG]
        reference = get(references, display_name(dataset), nothing)
        complete = all(s -> any(r -> Symbol(r["strategy"]) == s, group), STRATEGIES)
        X, y = if complete && reference !== nothing
            nothing, nothing
        else
            prepare_problem(dataset)
        end
        changed = false
        if reference === nothing
            GC.gc()
            fit = cdsolver(
                X, y, REG;
                lossfun = LogisticLoss(), intercept_strategy = NewtonStrategy(),
                tol = REFERENCE_TOL, maxit = 10_000, randomize = false
            )
            reference = fitted_diagnostics(dataset, X, y, fit)
            if !any(r -> Symbol(r["strategy"]) == :newton, group)
                push!(group, trajectory(dataset, :newton, fit))
                changed = true
            end
        end
        check_reference(dataset, reference, group)
        push!(rows, reference)
        for strategy in STRATEGIES
            if any(r -> Symbol(r["strategy"]) == strategy, group)
                println("$dataset $strategy (cached)")
                continue
            end
            println("$dataset $strategy")
            flush(stdout)
            intercept_strategy = if strategy == :gradient
                GradientStrategy()
            elseif strategy == :newton
                NewtonStrategy()
            else
                ExactStrategy()
            end
            GC.gc()
            fit = cdsolver(
                X, y, REG;
                lossfun = LogisticLoss(), intercept_strategy = intercept_strategy,
                tol = REFERENCE_TOL, maxit = 1000,
                maxtime = dataset == "news20-imb" ? 120.0 : Inf,
                randomize = false
            )
            push!(group, trajectory(dataset, strategy, fit))
            changed = true
        end
        if changed
            for r in group
                if haskey(r, "dual_relgaps")
                    r["relgaps"] = copy(r["dual_relgaps"])
                end
            end
            suboptimality_against_shared_dual!(group; instance_of = r -> (r["dataset"], r["reg"]))
        end
        for r in group
            for key in (:n, :p, :pi_plus, :intercept_only_fraction, :H00_over_L0)
                r[string(key)] = getproperty(reference, key)
            end
        end
        append!(results, group)
        println("  H00/L0=$(reference.H00_over_L0)")
        flush(stdout)
    end

    # Replace atomically so a preview cannot observe a partly written cache.
    staging = tempname(dirname(CACHE))
    jldsave(staging; results)
    mv(staging, CACHE; force = true)
    CSV.write(DIAGNOSTICS, DataFrame(rows))
    println("Saved $(length(results)) trajectories for $(length(DATASETS)) datasets")
    return nothing
end

main()
