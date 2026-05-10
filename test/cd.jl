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
        intercept_strategy = ConvergenceStrategy(),
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

    @test isapprox(res_grad.coef, res_newt.coef; atol = 1e-4)
    @test isapprox(res_grad.coef, res_conv.coef; atol = 1e-4)
    @test isapprox(res_grad.coef, res_damp.coef; atol = 1e-4)
    @test isapprox(res_grad.intercept, res_conv.intercept; atol = 1e-4)
    @test isapprox(res_grad.intercept, res_newt.intercept; atol = 1e-4)
    @test isapprox(res_grad.intercept, res_damp.intercept; atol = 1e-4)
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
