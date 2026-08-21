using Test
using Intercepts
using GLM

@testset "Intercepts methods converge" begin
    n = 100
    p = 10
    k = 2

    Random.seed!(1234)

    μ0 = 0.9

    X, y = generatedata(
        n,
        p;
        response = :binomial,
        μ0 = μ0,
        x_type = :normal,
        x_density = 0.9,
        ρ = 0.3,
        s = k,
        amplitude = 1,
    )

    reg = 0.01
    maxit = 1000

    res_grad = cdsolver(
        X,
        y,
        reg,
        lossfun = LogisticLoss(),
        intercept_strategy = GradientStrategy(),
        maxit = maxit,
    )
    res_newt = cdsolver(
        X,
        y,
        reg,
        lossfun = LogisticLoss(),
        intercept_strategy = NewtonStrategy(),
        maxit = maxit,
    )
    res_conv = cdsolver(
        X,
        y,
        reg,
        lossfun = LogisticLoss(),
        intercept_strategy = ExactStrategy(),
        maxit = maxit,
    )
    res_btgrad = cdsolver(
        X,
        y,
        reg,
        lossfun = LogisticLoss(),
        intercept_strategy = BacktrackingGradientStrategy(),
        maxit = maxit,
    )

    @test isapprox(res_grad.coef, res_newt.coef; atol = 1.0e-4)
    @test isapprox(res_grad.coef, res_conv.coef; atol = 1.0e-4)
    @test isapprox(res_grad.coef, res_btgrad.coef; atol = 1.0e-4)
    @test isapprox(res_grad.intercept, res_conv.intercept; atol = 1.0e-4)
    @test isapprox(res_grad.intercept, res_newt.intercept; atol = 1.0e-4)
    @test isapprox(res_grad.intercept, res_btgrad.intercept; atol = 1.0e-4)
end

@testset "Warm-starting from the optimum converges immediately" begin
    Random.seed!(4242)
    n, p, k = 100, 10, 2

    X, y = generatedata(
        n,
        p;
        response = :binomial,
        μ0 = 0.8,
        x_type = :normal,
        ρ = 0.3,
        s = k,
        amplitude = 1.0,
    )

    reg = 0.05
    res_cold = cdsolver(
        X,
        y,
        reg;
        lossfun = LogisticLoss(),
        intercept_strategy = NewtonStrategy(),
        maxit = 2000,
        tol = 1.0e-12,
    )

    res_warm = cdsolver(
        X,
        y,
        reg;
        lossfun = LogisticLoss(),
        intercept_strategy = NewtonStrategy(),
        maxit = 2000,
        tol = 1.0e-10,
        coef_init = res_cold.coef,
        intercept_init = res_cold.intercept,
    )

    @test res_warm.passes == 1
    @test res_warm.relgaps[1] < 1.0e-10
    @test isapprox(res_warm.coef, res_cold.coef; atol = 1.0e-8)
    @test isapprox(res_warm.intercept, res_cold.intercept; atol = 1.0e-8)
end

@testset "coef_init=nothing matches default zero-init" begin
    Random.seed!(99)
    n, p = 80, 12
    X, y = generatedata(
        n,
        p;
        response = :binomial,
        μ0 = 0.7,
        x_type = :normal,
        ρ = 0.2,
        s = 3,
        amplitude = 1.0,
    )

    res_default = cdsolver(
        X,
        y,
        0.1;
        lossfun = LogisticLoss(),
        intercept_strategy = NewtonStrategy(),
        maxit = 500,
    )
    res_explicit = cdsolver(
        X,
        y,
        0.1;
        lossfun = LogisticLoss(),
        intercept_strategy = NewtonStrategy(),
        maxit = 500,
        coef_init = zeros(p),
        intercept_init = 0.0,
    )

    @test isapprox(res_default.coef, res_explicit.coef; atol = 1.0e-8)
    @test isapprox(res_default.intercept, res_explicit.intercept; atol = 1.0e-8)
end

@testset "Per-coord Armijo gives monotone primal" begin
    Random.seed!(7777)
    n, p, k = 200, 50, 5

    X, y = generatedata(
        n,
        p;
        response = :binomial,
        μ0 = 0.9,
        x_type = :normal,
        ρ = 0.3,
        s = k,
        amplitude = 1.0,
    )

    strategies = [
        GradientStrategy(),
        NewtonStrategy(),
        ExactStrategy(),
        BacktrackingGradientStrategy(),
    ]

    for s in strategies
        res = cdsolver(
            X,
            y,
            0.05;
            lossfun = LogisticLoss(),
            intercept_strategy = s,
            maxit = 500,
            randomize = false,
            tol = 1.0e-12,
        )
        @test all(diff(res.primals) .<= 1.0e-10)
        @test isfinite(res.primals[end])
    end
end

@testset "NewtonStrategy intercept is descent at extreme imbalance" begin
    Random.seed!(8888)
    n = 200
    p = 50

    X, y = generatedata(
        n,
        p;
        response = :binomial,
        μ0 = 0.99,
        x_type = :normal,
        ρ = 0.3,
        s = 5,
        amplitude = 1.0,
    )

    # Construct an η that an unguarded Newton step would overshoot on:
    # mostly negative (matching mu0 near 1) with the intercept far from optimal.
    η = -3.0 .* ones(n) .+ 0.1 .* randn(n)
    f = LogisticLoss()

    f0 = loss(f, η, y)
    intercept_new = update_intercept(NewtonStrategy(), f, 0.0, η, y)
    η_new = η .+ (intercept_new - 0.0)
    f_new = loss(f, η_new, y)

    @test f_new <= f0 + 1.0e-10
end

@testset "ExactStrategy safeguards an extreme Poisson cold start" begin
    n = 100
    y = fill(300.0, n)
    η = zeros(n)
    f = PoissonLoss()

    intercept_new = update_intercept(ExactStrategy(), f, 0.0, η, y)
    η_new = η .+ intercept_new
    normalized_gradient = abs(sum(gradient(f, η_new, y))) / n

    @test isapprox(intercept_new, log(300.0); atol = 1.0e-10)
    @test normalized_gradient <= 1.0e-10
    @test loss(f, η_new, y) < loss(f, η, y)
end

@testset "ExactStrategy does not silently return a capped iterate" begin
    y = fill(300.0, 10)
    η = zeros(10)

    @test_throws ErrorException update_intercept(
        ExactStrategy(maxit = 1),
        PoissonLoss(),
        0.0,
        η,
        y,
    )
end

@testset "UnguardedNewtonStrategy matches Newton on quadratic loss" begin
    Random.seed!(2024)
    n = 200
    p = 20

    X, y = generatedata(
        n,
        p;
        response = :normal,
        x_type = :normal,
        ρ = 0.3,
        s = 5,
        amplitude = 1.0,
    )

    reg = 0.05
    res_newt = cdsolver(
        X,
        y,
        reg,
        lossfun = QuadraticLoss(),
        intercept_strategy = NewtonStrategy(),
        maxit = 500,
    )
    res_ung = cdsolver(
        X,
        y,
        reg,
        lossfun = QuadraticLoss(),
        intercept_strategy = UnguardedNewtonStrategy(),
        maxit = 500,
    )

    # Quadratic loss: the Armijo accept condition holds at α = 1 trivially
    # (the linear model is the true loss), so the unguarded and guarded
    # variants must produce identical iterates.
    @test isapprox(res_newt.coef, res_ung.coef; atol = 1.0e-6)
    @test isapprox(res_newt.intercept, res_ung.intercept; atol = 1.0e-6)
end

@testset "UnguardedNewtonStrategy vs Newton at extreme imbalance cold start" begin
    Random.seed!(8888)
    n = 200
    p = 50

    X, y = generatedata(
        n,
        p;
        response = :binomial,
        μ0 = 0.99,
        x_type = :normal,
        ρ = 0.3,
        s = 5,
        amplitude = 1.0,
    )

    # Same adversarial cold-start η as the descent test above.
    η = -3.0 .* ones(n) .+ 0.1 .* randn(n)
    f = LogisticLoss()

    f0 = loss(f, η, y)
    intercept_newt = update_intercept(NewtonStrategy(), f, 0.0, η, y)
    intercept_ung = update_intercept(UnguardedNewtonStrategy(), f, 0.0, η, y)

    η_newt = η .+ intercept_newt
    η_ung = η .+ intercept_ung
    f_newt = loss(f, η_newt, y)
    f_ung = loss(f, η_ung, y)

    # The guarded variant must descend (asserted in the earlier testset, repeated
    # here for the side-by-side narrative). The unguarded variant may or may not
    # descend on a given regime — what we want to record is the gap between the
    # two in this adversarial cold start.
    @test f_newt <= f0 + 1.0e-10
    @info "Adversarial cold start ΔF" guarded = f_newt - f0 unguarded = f_ung - f0
end

@testset "LogisticLoss stays finite under saturation" begin
    f = LogisticLoss()

    # Naive log1p(exp(η)) overflows to Inf for η ≳ 709; the loss must not.
    η = [1000.0, -1000.0, 50.0, 0.0]
    y = [1.0, 0.0, 1.0, 1.0]
    L = loss(f, η, y)
    @test isfinite(L)
    # softplus(η) - η y per observation: (1000-1000) + (0-0) + (50-50) + (log2-0).
    @test isapprox(L, log(2); atol = 1.0e-9)

    # Where the naive form does not overflow, the stable form agrees with it.
    ηs = [-8.0, -1.0, 0.0, 2.0, 9.0]
    ys = [0.0, 1.0, 0.0, 1.0, 0.0]
    @test isapprox(loss(f, ηs, ys), sum(log1p.(exp.(ηs)) .- ηs .* ys); atol = 1.0e-12)
end
