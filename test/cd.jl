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
    res_damp = cdsolver(
        X,
        y,
        reg,
        lossfun = LogisticLoss(),
        intercept_strategy = DampedNewtonStrategy(),
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

    @test isapprox(res_grad.coef, res_newt.coef; atol = 1e-4)
    @test isapprox(res_grad.coef, res_conv.coef; atol = 1e-4)
    @test isapprox(res_grad.coef, res_damp.coef; atol = 1e-4)
    @test isapprox(res_grad.coef, res_btgrad.coef; atol = 1e-4)
    @test isapprox(res_grad.intercept, res_conv.intercept; atol = 1e-4)
    @test isapprox(res_grad.intercept, res_newt.intercept; atol = 1e-4)
    @test isapprox(res_grad.intercept, res_damp.intercept; atol = 1e-4)
    @test isapprox(res_grad.intercept, res_btgrad.intercept; atol = 1e-4)
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
        tol = 1e-12,
    )

    res_warm = cdsolver(
        X,
        y,
        reg;
        lossfun = LogisticLoss(),
        intercept_strategy = NewtonStrategy(),
        maxit = 2000,
        tol = 1e-10,
        coef_init = res_cold.coef,
        intercept_init = res_cold.intercept,
    )

    @test res_warm.passes == 1
    @test res_warm.relgaps[1] < 1e-10
    @test isapprox(res_warm.coef, res_cold.coef; atol = 1e-8)
    @test isapprox(res_warm.intercept, res_cold.intercept; atol = 1e-8)
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

    @test isapprox(res_default.coef, res_explicit.coef; atol = 1e-8)
    @test isapprox(res_default.intercept, res_explicit.intercept; atol = 1e-8)
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
        DampedNewtonStrategy(),
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
            tol = 1e-12,
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

@testset "DampedNewton matches Newton on quadratic" begin
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
    res_damp = cdsolver(
        X,
        y,
        reg,
        lossfun = QuadraticLoss(),
        intercept_strategy = DampedNewtonStrategy(),
        maxit = 500,
    )

    @test isapprox(res_newt.coef, res_damp.coef; atol = 1e-6)
    @test isapprox(res_newt.intercept, res_damp.intercept; atol = 1e-6)
end
