using Test
using Intercepts
using LinearAlgebra
using Random

@testset "MultinomialLogisticLoss numerical sanity" begin
    Random.seed!(1)

    # Tiny problem: K=3, n=4
    K = 3
    n = 4
    f = MultinomialLogisticLoss(K = K)
    η = randn(n, K - 1)
    y = [1, 2, 3, 1]

    p = softmax_probs(η)

    @test size(p) == (n, K)
    @test all(sum(p; dims = 2) .≈ 1.0)
    @test all(p .>= 0)

    # Reference-class K invariance: η_iK = 0 implicit
    p2 = softmax_probs(η .+ 0) # no shift, same input
    @test p ≈ p2

    # Compare loss to direct cross-entropy via softmax_probs
    L = loss(f, η, y)
    L_ref = -sum(log(p[i, y[i]]) for i in 1:n)
    @test L ≈ L_ref

    # Finite-difference check on gradient
    ε = 1.0e-6
    G = gradient(f, η, y)
    @test size(G) == (n, K - 1)
    for i in 1:n, k in 1:(K - 1)
        ηp = copy(η)
        ηm = copy(η)
        ηp[i, k] += ε
        ηm[i, k] -= ε
        g_fd = (loss(f, ηp, y) - loss(f, ηm, y)) / (2ε)
        @test isapprox(G[i, k], g_fd; atol = 1.0e-6)
    end

    # Finite-difference check on Hessian (per-observation block)
    H = hessian(f, η, y)
    @test size(H) == (n, K - 1, K - 1)
    for i in 1:n, k in 1:(K - 1), l in 1:(K - 1)
        ηp = copy(η)
        ηm = copy(η)
        ηp[i, l] += ε
        ηm[i, l] -= ε
        Gp = gradient(f, ηp, y)
        Gm = gradient(f, ηm, y)
        h_fd = (Gp[i, k] - Gm[i, k]) / (2ε)
        @test isapprox(H[i, k, l], h_fd; atol = 1.0e-6)
    end
end

@testset "MultinomialLogisticLoss reduces to LogisticLoss for K=2" begin
    Random.seed!(2)
    n = 50

    # Generate η as scalar; embed as Matrix(n, 1) for multinomial.
    η_scalar = randn(n)
    η_mat = reshape(η_scalar, n, 1)

    # Binomial y in {0, 1}; multinomial y in {1, 2}.
    y_bin = rand([0.0, 1.0], n)
    y_mult = Int.(2 .- y_bin) # y_bin=1 (success) → class 1; y_bin=0 → class 2

    fbin = LogisticLoss()
    fmult = MultinomialLogisticLoss(K = 2, lipschitz = 0.25)

    @test isapprox(loss(fbin, η_scalar, y_bin), loss(fmult, η_mat, y_mult); atol = 1.0e-10)

    G_bin = gradient(fbin, η_scalar, y_bin)
    G_mult = gradient(fmult, η_mat, y_mult)
    @test isapprox(G_bin, vec(G_mult); atol = 1.0e-10)

    H_bin = hessian(fbin, η_scalar, y_bin)
    H_mult = hessian(fmult, η_mat, y_mult)
    @test isapprox(H_bin, [H_mult[i, 1, 1] for i in 1:n]; atol = 1.0e-10)
end

@testset "Vector-intercept strategies reduce to scalar on K=2" begin
    Random.seed!(3)
    n = 50

    η_scalar = randn(n)
    η_mat = reshape(η_scalar, n, 1)
    y_bin = rand([0.0, 1.0], n)
    y_mult = Int.(2 .- y_bin)

    fbin = LogisticLoss()
    fmult = MultinomialLogisticLoss(K = 2, lipschitz = 0.25)

    β0_scalar = 0.5
    β0_vec = [0.5]

    # NoIntercept
    @test update_intercept(NoIntercept(), fbin, β0_scalar, η_scalar, y_bin) == 0
    @test update_intercept(NoIntercept(), fmult, β0_vec, η_mat, y_mult) == [0.0]

    # GradientStrategy
    s_scalar = update_intercept(GradientStrategy(), fbin, β0_scalar, η_scalar, y_bin)
    s_vec = update_intercept(GradientStrategy(), fmult, β0_vec, η_mat, y_mult)
    @test isapprox(s_scalar, s_vec[1]; atol = 1.0e-12)

    # NewtonStrategy
    s_scalar = update_intercept(NewtonStrategy(), fbin, β0_scalar, η_scalar, y_bin)
    s_vec = update_intercept(NewtonStrategy(), fmult, β0_vec, η_mat, y_mult)
    @test isapprox(s_scalar, s_vec[1]; atol = 1.0e-12)

    # ExactStrategy
    s_scalar = update_intercept(ExactStrategy(), fbin, β0_scalar, η_scalar, y_bin)
    s_vec = update_intercept(ExactStrategy(), fmult, β0_vec, η_mat, y_mult)
    @test isapprox(s_scalar, s_vec[1]; atol = 1.0e-10)
end

@testset "Multinomial per-coord Armijo gives monotone primal" begin
    Random.seed!(5151)
    n = 200
    p = 40
    K = 3

    X, y = generatedata(
        n,
        p;
        response = :multinomial,
        K = K,
        class_probs = [0.85, 0.1, 0.05],
        x_type = :normal,
        ρ = 0.3,
        s = 4,
        amplitude = 1.0,
    )

    for s in (NewtonStrategy(), ExactStrategy(), GradientStrategy())
        res = multinomial_cdsolver(
            X,
            y,
            0.05;
            lossfun = MultinomialLogisticLoss(K = K),
            intercept_strategy = s,
            maxit = 200,
            randomize = false,
            tol = 1.0e-12,
        )
        @test all(diff(res.primals) .<= 1.0e-10)
        @test isfinite(res.primals[end])
    end
end

@testset "multinomial_cdsolver K=2 reduces to cdsolver" begin
    Random.seed!(4)
    n = 200
    p = 30

    X, y_bin = generatedata(
        n,
        p;
        response = :binomial,
        μ0 = 0.4,
        x_type = :normal,
        ρ = 0.3,
        s = 5,
        amplitude = 1.0,
    )
    y_mult = Int.(2 .- y_bin) # y_bin=1 → class 1, y_bin=0 → class 2

    reg = 0.05
    maxit = 500

    # Newton (deterministic): should match exactly across binomial vs K=2 multinomial.
    res_bin = cdsolver(
        X,
        y_bin,
        reg;
        lossfun = LogisticLoss(),
        intercept_strategy = NewtonStrategy(),
        maxit = maxit,
        randomize = false,
        tol = 1.0e-12,
    )
    res_mult = multinomial_cdsolver(
        X,
        y_mult,
        reg;
        lossfun = MultinomialLogisticLoss(K = 2, lipschitz = 0.25),
        intercept_strategy = NewtonStrategy(),
        maxit = maxit,
        randomize = false,
        tol = 1.0e-12,
    )

    @test isapprox(res_bin.coef, vec(res_mult.coef); atol = 1.0e-4)
    @test isapprox(res_bin.intercept, res_mult.intercept[1]; atol = 1.0e-4)
end

@testset "Multinomial dual is a valid optimum certificate" begin
    Random.seed!(909)
    n, p, K = 300, 40, 4
    X, y = generatedata(
        n,
        p;
        response = :multinomial,
        K = K,
        class_probs = [0.6, 0.2, 0.15, 0.05],
        x_type = :normal,
        ρ = 0.3,
        s = 4,
        amplitude = 1.0,
    )
    res = multinomial_cdsolver(
        X,
        y,
        0.05;
        lossfun = MultinomialLogisticLoss(K = K),
        intercept_strategy = ExactStrategy(),
        maxit = 500,
        randomize = false,
        tol = 1.0e-12,
    )

    scale = max.(abs.(res.primals), 1.0)
    # Weak duality: the dual point lower-bounds the primal, so the gap is
    # nonnegative up to floating-point noise.
    @test all(res.gaps .>= -1.0e-9 .* scale)
    @test all(res.duals .<= res.primals .+ 1.0e-9 .* scale)
    # The gap shrinks enough to certify the iterate as a near-optimum, which is
    # how the experiments establish a trustworthy F* for primal-suboptimality
    # plots. It need not reach machine zero: the dual point is a feasible
    # lower-bound construction, not the exact dual optimum.
    @test minimum(res.relgaps) < 1.0e-4
end

@testset "Shared multinomial dual gives a certified suboptimality bound" begin
    results = Any[
        Dict{String, Any}(
            "primals" => [1.2, 1.05],
            "duals" => [0.7, 0.99],
            "relgaps" => [0.5, 0.1],
        ),
        Dict{String, Any}(
            "primals" => [1.1, 1.0],
            "duals" => [0.8, 0.98],
            "relgaps" => [0.3, 0.02],
        ),
    ]

    suboptimality_against_certified_optimum!(
        results;
        instance_of = _ -> 1,
        cert_tol = 0.02,
    )

    @test all(r["dual_bound"] == 0.99 for r in results)
    @test all(r["Fstar"] == 1.0 for r in results)
    @test results[2]["gaps"] ≈ [0.11, 0.01]
    @test results[2]["relgaps"] ≈ [0.11, 0.01]
    @test results[2]["primal_relgaps"] ≈ [0.1, 1.0e-20]
end

@testset "Multinomial K=2 dual matches binary logistic dual" begin
    Random.seed!(77)
    n = 60
    η_scalar = randn(n)
    y_bin = rand([0.0, 1.0], n)
    y_mult = Int.(2 .- y_bin)

    fbin = LogisticLoss()
    fmult = MultinomialLogisticLoss(K = 2, lipschitz = 0.25)

    # Dual point: the binary residual scaled toward the interior so the implied
    # probabilities stay strictly inside the (here one-dimensional) simplex.
    θ = residual(fbin, η_scalar, y_bin) ./ 1.5

    @test isapprox(
        dual(fbin, θ, y_bin),
        dual(fmult, reshape(θ, n, 1), y_mult);
        atol = 1.0e-10,
    )
end
