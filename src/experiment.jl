using LIBSVMdata

function get_lossfun(response)
    if response == :binomial
        return Logistic()
    else
        error("Unknown response type: $response")
    end
end

function get_intercept_strategy(strategy)
    if strategy == :gradient
        return GradientStrategy()
    elseif strategy == :newton
        return NewtonStrategy()
    elseif strategy == :convergence
        return ConvergenceStrategy()
    else
        error("Unknown strategy: $strategy")
    end
end

function run_solver(X, y; response, strategy, reg, maxit, maxtime, randomize, update_freq)
    lossfun = get_lossfun(response)
    intercept_strategy = get_intercept_strategy(strategy)

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
        y .= Int.(y .== 1)
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
