using Random
using SparseArrays

function get_lossfun(response; K::Int = 3, lipschitz::Union{Real, Nothing} = nothing)
    if response == :binomial
        return LogisticLoss()
    elseif response == :gaussian
        return QuadraticLoss()
    elseif response == :poisson
        return PoissonLoss()
    elseif response == :multinomial
        L = lipschitz === nothing ? 0.5 : Float64(lipschitz)
        return MultinomialLogisticLoss(K = K, lipschitz = L)
    else
        error("Unknown response type: $response")
    end
end

function get_intercept_strategy(strategy)
    if strategy == :gradient
        return GradientStrategy()
    elseif strategy == :newton
        return NewtonStrategy()
    elseif strategy == :unguarded_newton
        return UnguardedNewtonStrategy()
    elseif strategy == :bt_gradient
        return BacktrackingGradientStrategy()
    elseif strategy == :exact
        return ExactStrategy()
    else
        error("Unknown strategy: $strategy")
    end
end

function run_solver(
    X,
    y;
    response,
    strategy,
    reg,
    maxit,
    maxtime,
    randomize,
    update_freq,
    K::Int = 3,
)
    intercept_strategy = get_intercept_strategy(strategy)

    if response == :multinomial
        lossfun = get_lossfun(response; K = K)
        res = multinomial_cdsolver(
            X,
            y,
            reg;
            lossfun = lossfun,
            intercept_strategy = intercept_strategy,
            maxit = maxit,
            maxtime = maxtime,
            randomize = randomize,
            update_freq = update_freq,
        )
        # The multinomial solver now maintains a dual-feasible point by dual
        # scaling, so report the true relative duality gap (floored for the
        # log-scale plots) rather than a run-local primal-suboptimality proxy.
        return (
            time = res.time,
            gaps = res.gaps,
            relgaps = max.(res.relgaps, 1.0e-20),
            primals = res.primals,
            duals = res.duals,
        )
    end

    lossfun = get_lossfun(response)
    res = cdsolver(
        X,
        y,
        reg;
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
end

# Re-express each multinomial run's suboptimality as relative *primal*
# suboptimality against a shared optimum F*, the best primal achieved by any
# strategy on the same problem instance. The multinomial CD solver maintains a
# duality gap (the per-iteration `relgaps`), and we use it the way a referee
# would: confirm that at least one strategy drove the gap to ~0 --- certifying
# that F* is the true optimum --- then measure every run against that F*. This
# avoids the run-local-minimum understatement (a stalling run is scored against
# the shared optimum, not its own floor) and avoids reporting the raw gap, which
# can sit slightly above zero when a converged primal has a not-yet-tight dual
# point. `instance_of` maps a result dict to the key identifying its problem
# instance; results sharing a key share an F*. The original gap is preserved as
# `dual_relgaps`, and the certifying gap as `dual_cert`.
"""
    suboptimality_against_certified_optimum!(results; instance_of, cert_tol=1e-4)

Replace each run's gap history with relative primal suboptimality against the
best certified objective value among runs of the same problem instance.
`instance_of(result)` supplies the grouping key. The function preserves the
original relative duality gaps in `"dual_relgaps"` and records the shared
certificate and objective in `"dual_cert"` and `"Fstar"`.

The function mutates and returns `results`. It warns when no run in a group has
a relative duality gap at or below `cert_tol`.
"""
function suboptimality_against_certified_optimum!(
    results;
    instance_of::Function,
    cert_tol::Real = 1.0e-4,
)
    groups = Dict{Any, Vector{Any}}()
    for r in results
        push!(get!(groups, instance_of(r), Any[]), r)
    end
    for (key, group) in groups
        fstar = minimum(minimum(r["primals"]) for r in group)
        cert = minimum(minimum(r["relgaps"]) for r in group)
        cert <= cert_tol ||
            @warn "F* only weakly certified by the duality gap" instance = key gap = cert
        for r in group
            r["dual_relgaps"] = r["relgaps"]
            r["dual_cert"] = cert
            r["Fstar"] = fstar
            r["gaps"] = r["primals"] .- fstar
            r["relgaps"] = max.(
                (r["primals"] .- fstar) ./ max(abs(fstar), 1.0e-15),
                1.0e-20,
            )
        end
    end
    return results
end

"""
    simulated_experiment(n=100, p=10000; kwargs...)

Generate a synthetic GLM problem and run the coordinate-descent solver used by
the paper's experiment scripts. The `response` keyword selects `:binomial`,
`:gaussian`, `:poisson`, or `:multinomial`; `strategy` selects the intercept
update. Remaining keywords control data generation, regularization, coordinate
ordering, and solver budgets.

Return a named tuple containing time, gap, relative-gap, primal, and dual
histories.
"""
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
    K::Int = 3,
    class_probs::Union{AbstractVector, Nothing} = nothing,
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
        K = K,
        class_probs = class_probs,
    )

    return run_solver(
        X,
        y;
        response = response,
        strategy = strategy,
        reg = reg,
        maxit = maxit,
        maxtime = maxtime,
        randomize = randomize,
        update_freq = update_freq,
        K = K,
    )
end;

"""
    real_experiment(dataset="w1a"; kwargs...)

Load a staged dataset and run the coordinate-descent solver used by the paper's
real-data experiment scripts. For binary responses, the largest label is mapped
to the positive class. `n_positive` and `n_negative` optionally construct a
seeded class-imbalanced subset; `min_nnz_per_column` removes features without
enough retained support.

Return a named tuple containing time, gap, relative-gap, primal, and dual
histories.
"""
function real_experiment(
    dataset = "w1a";
    response = :binomial,
    strategy = :gradient,
    reg = 0.01,
    randomize = false,
    update_freq = 1,
    maxit = 1000,
    maxtime = Inf,
    n_positive::Union{Nothing, Int} = nothing,
    n_negative::Union{Nothing, Int} = nothing,
    seed::Int = 42,
    min_nnz_per_column::Int = 1,
)

    X, y = load_local_dataset(dataset)

    if response == :binomial
        # LIBSVM binary datasets ship in several encodings ({-1, 1}, {0, 1},
        # {2, 4}, ...); map the max-valued class to the positive class so
        # the conversion is encoding-agnostic.
        pos = maximum(y)
        y .= Int.(y .== pos)
    end

    if n_positive !== nothing || n_negative !== nothing
        # Subsample rows to create a controlled class imbalance. Drop any
        # feature column that is too sparse to standardise stably (sparse
        # text features with one or two surviving nonzeros after row
        # subsampling otherwise standardise to extreme values and destabilise
        # the CD solve).
        rng = MersenneTwister(seed)
        pos_idx = findall(==(1), y)
        neg_idx = findall(==(0), y)
        n_pos_keep = n_positive === nothing ?
            length(pos_idx) :
            min(n_positive, length(pos_idx))
        n_neg_keep = n_negative === nothing ?
            length(neg_idx) :
            min(n_negative, length(neg_idx))
        pos_sample = shuffle(rng, pos_idx)[1:n_pos_keep]
        neg_sample = shuffle(rng, neg_idx)[1:n_neg_keep]
        row_idx = sort(vcat(pos_sample, neg_sample))
        X = X[row_idx, :]
        y = y[row_idx]

        # For sparse CSC matrices, nnz/column reads off the colptr in O(p)
        # without materialising per-column views. min_nnz_per_column = 1
        # only drops fully-empty columns (preserves the prior contract);
        # higher values drop low-support features that otherwise produce
        # near-zero standard deviations and NaN gradients downstream.
        if issparse(X)
            Xc = X::SparseMatrixCSC
            keep = [
                Xc.colptr[j + 1] - Xc.colptr[j] >= min_nnz_per_column
                for j in 1:size(Xc, 2)
            ]
        else
            keep = [
                !iszero(maximum(@view X[:, j]) - minimum(@view X[:, j]))
                for j in axes(X, 2)
            ]
        end
        X = X[:, keep]
    end

    return run_solver(
        X,
        y;
        response = response,
        strategy = strategy,
        reg = reg,
        maxit = maxit,
        maxtime = maxtime,
        randomize = randomize,
        update_freq = update_freq,
    )
end;
