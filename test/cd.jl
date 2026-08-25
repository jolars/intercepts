using Test
using Intercepts
using GLM

@testset "Intercept update frequency" begin
    X = [1.0 0.0; 0.0 1.0; 1.0 1.0; -1.0 -1.0]
    y = [0.0, 0.0, 0.0, 1.0]

    res = cdsolver(X, y; lossfun = LogisticLoss(), update_freq = 3, maxit = 2)

    @test all(isfinite, res.primals)
    @test res.intercept != 0.0
    @test_throws ArgumentError cdsolver(
        X,
        y;
        lossfun = LogisticLoss(),
        update_freq = 0,
    )
end

@testset "Dual certificates respect conjugate domains" begin
    y_logistic = [1.0, 1.0, 1.0, 0.0]
    p_logistic = [0.99, 0.01, 0.01, 0.01]
    η_logistic = log.(p_logistic ./ (1 .- p_logistic))
    θ_logistic = Intercepts._dual_point(LogisticLoss(), η_logistic, y_logistic, true)
    u_logistic = y_logistic + θ_logistic
    raw_logistic = residual(LogisticLoss(), η_logistic, y_logistic)
    naive_logistic = raw_logistic .- mean(raw_logistic)

    @test any((y_logistic .+ naive_logistic .< 0) .| (y_logistic .+ naive_logistic .> 1))
    @test isapprox(sum(θ_logistic), 0.0; atol = 1.0e-14)
    @test all((0 .<= u_logistic) .& (u_logistic .<= 1))
    @test dual(LogisticLoss(), raw_logistic, y_logistic) ≈
        loss(LogisticLoss(), η_logistic, y_logistic) - sum(raw_logistic .* η_logistic)
    @test_throws DomainError dual(LogisticLoss(), [0.2], [1.0])
    @test_throws ArgumentError dual(LogisticLoss(), [NaN], [0.0])

    y_poisson = [0.0, 10.0, 10.0]
    μ_poisson = [0.01, 20.0, 20.0]
    η_poisson = log.(μ_poisson)
    θ_poisson = Intercepts._dual_point(PoissonLoss(), η_poisson, y_poisson, true)
    u_poisson = y_poisson + θ_poisson
    raw_poisson = residual(PoissonLoss(), η_poisson, y_poisson)
    naive_poisson = raw_poisson .- mean(raw_poisson)

    @test any(y_poisson .+ naive_poisson .< 0)
    @test isapprox(sum(θ_poisson), 0.0; atol = 1.0e-14)
    @test all(u_poisson .>= 0)
    @test dual(PoissonLoss(), raw_poisson, y_poisson) ≈
        loss(PoissonLoss(), η_poisson, y_poisson) - sum(raw_poisson .* η_poisson)
    @test_throws DomainError dual(PoissonLoss(), [-0.1], [0.0])
    @test_throws ArgumentError dual(PoissonLoss(), [NaN], [0.0])
end

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

@testset "Fixed primal target stops before a pass" begin
    Random.seed!(199)
    X, y = generatedata(40, 8; response = :binomial, μ0 = 0.7, s = 3)
    result = cdsolver(
        X,
        y,
        0.05;
        lossfun = LogisticLoss(),
        intercept_strategy = NewtonStrategy(),
        primal_stop = Inf,
    )

    @test length(result.primals) == 1
    @test isempty(result.inner_steps)
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

@testset "ExactStrategy cap uses its documented numerical margin" begin
    f = PoissonLoss()
    y = [1.0]
    tol = 1.0e-4
    strategy = ExactStrategy(gradient_tol = tol, maxit = 1)

    # These starts place the post-Newton residual on opposite sides of the
    # floating-point margin and make the cap policy observable in one step.
    η_within_margin = [0.02]
    intercept = update_intercept(strategy, f, 0.0, η_within_margin, y)
    normalized_gradient = abs(only(gradient(f, η_within_margin .+ intercept, y)))

    @test tol < normalized_gradient <= 5 * tol
    @test_throws ErrorException update_intercept(strategy, f, 0.0, [0.04], y)
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

@testset "Shared dual reference is strategy-independent" begin
    results = [
        Dict{String, Any}(
            "instance" => :a,
            "primals" => [10.0, 5.0],
            "duals" => [0.0, 4.0],
            "gaps" => [10.0, 1.0],
            "relgaps" => [1.0, 0.2],
        ),
        Dict{String, Any}(
            "instance" => :a,
            "primals" => [9.0, 4.5],
            "duals" => [1.0, 4.4],
            "gaps" => [8.0, 0.1],
            "relgaps" => [8 / 9, 1 / 45],
        ),
        Dict{String, Any}(
            "instance" => :a,
            "primals" => Float64[],
            "duals" => Float64[],
            "gaps" => Float64[],
            "relgaps" => Float64[],
        ),
        Dict{String, Any}(
            "instance" => :b,
            "primals" => [3.0],
            "duals" => [2.5],
            "gaps" => [0.5],
            "relgaps" => [1 / 6],
        ),
    ]

    suboptimality_against_shared_dual!(
        results;
        instance_of = r -> r["instance"],
        cert_tol = 1.0,
    )

    @test results[1]["dual_bound"] == 4.4
    @test results[2]["dual_bound"] == 4.4
    @test results[1]["Fstar"] == 4.5
    @test results[1]["gaps"] ≈ [5.6, 0.6]
    @test results[2]["gaps"] ≈ [4.6, 0.1]
    @test results[1]["dual_relgaps"] == [1.0, 0.2]
    @test results[2]["primal_relgaps"] ≈ [1.0, 1.0e-20]
    @test isempty(results[3]["relgaps"])
    @test results[3]["dual_bound"] == 4.4
    @test results[4]["dual_bound"] == 2.5
end
