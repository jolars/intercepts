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

    @test isapprox(res_grad.coef, res_newt.coef; atol = 1e-4)
    @test isapprox(res_grad.coef, res_conv.coef; atol = 1e-4)
    @test isapprox(res_grad.intercept, res_conv.intercept; atol = 1e-4)
    @test isapprox(res_grad.intercept, res_newt.intercept; atol = 1e-4)
end
