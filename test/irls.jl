using Test
using Intercepts
using Random
using LinearAlgebra
using Statistics

@testset "IRLS converges to same minimizer as cdsolver" begin
    Random.seed!(1)
    n, p = 200, 30
    X, y = generatedata(
        n,
        p;
        response = :binomial,
        μ0 = 0.7,
        x_type = :normal,
        ρ = 0.3,
        s = 5,
        amplitude = 1.0,
    )

    reg = 0.05

    res_cd = cdsolver(
        X,
        y,
        reg;
        lossfun = LogisticLoss(),
        intercept_strategy = NewtonStrategy(),
        maxit = 2000,
        tol = 1.0e-12,
        randomize = false,
    )
    res_irls = irlssolver(
        X,
        y,
        reg;
        lossfun = LogisticLoss(),
        weights = LocalWeights(),
        intercept_strategy = NewtonStrategy(),
        max_outer = 200,
        max_inner = 100,
        outer_tol = 1.0e-12,
        randomize = false,
    )

    @test isapprox(res_cd.coef, res_irls.coef; atol = 1.0e-5)
    @test isapprox(res_cd.intercept, res_irls.intercept; atol = 1.0e-5)
end

@testset "Local-weight IRLS strategy collapse" begin
    Random.seed!(2)
    n, p = 150, 20
    X, y = generatedata(
        n,
        p;
        response = :binomial,
        μ0 = 0.6,
        x_type = :normal,
        ρ = 0.3,
        s = 4,
        amplitude = 1.0,
    )
    reg = 0.05

    common = (
        lossfun = LogisticLoss(),
        weights = LocalWeights(),
        max_outer = 50,
        max_inner = 20,
        outer_tol = 1.0e-12,
        randomize = false,
        save_history = true,
    )

    res_grad = irlssolver(X, y, reg; intercept_strategy = GradientStrategy(), common...)
    res_newt = irlssolver(X, y, reg; intercept_strategy = NewtonStrategy(), common...)
    res_conv = irlssolver(X, y, reg; intercept_strategy = ConvergenceStrategy(), common...)

    # All three strategies coincide on the inner quadratic — iterates should match.
    @test res_grad.intercepts == res_newt.intercepts
    @test res_grad.intercepts == res_conv.intercepts
    @test res_grad.coefs == res_newt.coefs
    @test res_grad.coefs == res_conv.coefs
end

@testset "MM-IRLS first inner intercept step equals direct-CD GradientStrategy" begin
    Random.seed!(3)
    n = 100
    y_int = rand(0:1, n)
    y = Float64.(y_int)
    η = zeros(n)

    # Direct-CD GradientStrategy intercept update from (β_0=0, η=0):
    direct_step = update_intercept(GradientStrategy(), LogisticLoss(), 0.0, η, y)

    # Manual MM-IRLS inner intercept update from (β_0=0, β=0, η=0):
    w = fill(0.25, n)
    μ_vec = invlink(LogisticLoss(), η)
    z = η .+ (y .- μ_vec) ./ w
    grad_0 = sum(w .* (η .- z))
    curv = sum(w)
    mm_step = 0.0 - grad_0 / curv

    @test isapprox(direct_step, mm_step; atol = 1.0e-12)
end

@testset "Intercept-only logistic recovers logit(mean(y))" begin
    Random.seed!(4)
    n = 500
    y = Float64.(rand(0:1, n))

    # Zero-variance feature: standardize replaces the 0 stdev with 1, so β_1 has
    # no signal and stays at 0. The intercept is regularization-free (penalty
    # is on β only) and should converge to the unconditional MLE logit(mean(y)).
    X = zeros(n, 1)

    res = irlssolver(
        X,
        y,
        1.0e-6;
        lossfun = LogisticLoss(),
        weights = LocalWeights(),
        intercept_strategy = NewtonStrategy(),
        max_outer = 200,
        max_inner = 50,
        outer_tol = 1.0e-12,
    )

    expected = log(mean(y) / (1 - mean(y)))
    @test isapprox(res.intercept, expected; atol = 1.0e-6)
end

@testset "MajorizedWeights only supported for LogisticLoss (no Poisson)" begin
    # Solver signature already restricts lossfun to LogisticLoss, but MajorizedWeights
    # is the conceptual reason — exercise it explicitly via the weight helper.
    Random.seed!(5)
    n = 50
    η = zeros(n)
    y_count = Float64.(rand(0:5, n))

    @test_throws MethodError Intercepts._irls_weights(
        MajorizedWeights(),
        PoissonLoss(),
        η,
        y_count,
    )
end
