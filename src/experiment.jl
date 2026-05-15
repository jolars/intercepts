using LIBSVMdata

function get_lossfun(response; K::Int = 3, lipschitz::Union{Real,Nothing} = nothing)
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
    elseif strategy == :damped_newton
        return DampedNewtonStrategy()
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
        # No duality-gap path for multinomial; report primal suboptimality
        # against the run's minimum primal as a proxy for relgap.
        primals = res.primals
        pmin = minimum(primals)
        relgaps = max.((primals .- pmin) ./ max(abs(pmin), 1e-15), 1e-20)
        return (
            time = res.time,
            gaps = primals .- pmin,
            relgaps = relgaps,
            primals = primals,
            duals = fill(pmin, length(primals)),
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
    class_probs::Union{AbstractVector,Nothing} = nothing,
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


function real_experiment(
    dataset = "w1a";
    response = :binomial,
    strategy = :gradient,
    reg = 0.01,
    randomize = false,
    update_freq = 1,
    maxit = 1000,
    maxtime = Inf,
)

    X, y = load_dataset(dataset, verbose = false)

    if response == :binomial
        # LIBSVM binary datasets ship in several encodings ({-1, 1}, {0, 1},
        # {2, 4}, ...); map the max-valued class to the positive class so
        # the conversion is encoding-agnostic.
        pos = maximum(y)
        y .= Int.(y .== pos)
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
