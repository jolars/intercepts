using Test
using Intercepts
using LinearAlgebra
using SparseArrays
using Statistics

@testset "Sparse gradient descent uses the squared spectral norm" begin
    X = [4.0 0.0 0.0; -4.0 0.0 0.0; 0.0 3.0 0.0; 0.0 -3.0 0.0; 0.0 0.0 2.0; 0.0 0.0 -2.0]
    y = X * [1.2, -0.7, 0.4] .+ 1.3
    reg = 0.05

    dense = gdsolver(X, y, reg; normalization = :none, maxit = 30, tol = 0.0)
    sparse_fit = gdsolver(sparse(X), y, reg; normalization = :none, maxit = 30, tol = 0.0)

    @test sparse_fit.primals ≈ dense.primals
    @test sparse_fit.coef ≈ dense.coef
    @test sparse_fit.intercept ≈ dense.intercept
    @test all(diff(sparse_fit.primals) .<= 1.0e-12)

    converged = gdsolver(sparse(X), y, reg; normalization = :none, maxit = 1000, tol = 1.0e-12)
    λ = reg * lambdamax(QuadraticLoss(), X, y)
    expected_coef = sign.(X' * y) .* max.(abs.(X' * y) .- λ, 0.0) ./ vec(sum(abs2, X; dims = 1))
    expected_primal = loss(QuadraticLoss(), X * expected_coef .+ mean(y), y) + λ * norm(expected_coef, 1)

    @test converged.passes < 1000
    @test converged.gaps[end] / converged.primals[end] <= 1.0e-12
    @test converged.coef ≈ expected_coef atol = 1.0e-6
    @test converged.primals[end] ≈ expected_primal
end

@testset "Sparse gradient descent accepts one row or column" begin
    for X in (reshape([3.0, -1.0, 2.0], :, 1), [3.0 -1.0 2.0])
        y = collect(1.0:size(X, 1))
        dense = gdsolver(X, y; normalization = :none, intercept_strategy = NoIntercept())
        sparse_fit = gdsolver(sparse(X), y; normalization = :none, intercept_strategy = NoIntercept())

        @test sparse_fit.coef ≈ dense.coef
        @test sparse_fit.primals ≈ dense.primals
        @test sparse_fit.passes < 1000
        @test all(isfinite, sparse_fit.primals)
    end
end

@testset "Zero features leave an intercept-only fit" begin
    y = [1.0, 3.0, 2.0]
    for X in (zeros(3, 2), spzeros(3, 2))
        fit = gdsolver(X, y; normalization = :none)

        @test fit.coef == zeros(2)
        @test fit.intercept ≈ mean(y)
        @test fit.primals[end] ≈ loss(QuadraticLoss(), fill(mean(y), length(y)), y)
        @test fit.passes < 1000
    end
end
