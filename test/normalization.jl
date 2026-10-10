using Test
using Intercepts
using LinearAlgebra
using SparseArrays
using Statistics

@testset "Scale-only normalization preserves feature origins" begin
    X = [0.0 2.0; 1.0 2.0; 3.0 2.0; 0.0 2.0]
    scales_expected = [std(X[:, 1]; corrected = false) 1.0]
    centered, _, _ = normalizefeatures(X)

    @test centered ≈ (X .- mean(X; dims = 1)) ./ scales_expected
    for features in (X, sparse(X))
        original = copy(features)
        scaled, offsets, scales = normalizefeatures(features; center = false)
        @test scaled ≈ X ./ scales_expected
        @test offsets == zeros(1, 2)
        @test scales ≈ scales_expected
        @test features == original
        @test issparse(scaled) == issparse(features)
    end
    sparse_scaled, sparse_centers, sparse_scales = normalizefeatures(sparse(X))
    @test Matrix(sparse_scaled) .- sparse_centers ./ sparse_scales ≈ centered
end

@testset "NoIntercept quadratic fits match the analytic lasso solution" begin
    X = reshape([1.0, 2.0, 3.0, 4.0], :, 1)
    y = 2 .* vec(X)
    scale = std(vec(X); corrected = false)
    reg = 0.1
    λ = reg * abs(dot(vec(X) ./ scale, mean(y) .- y))
    expected = (dot(vec(X), y) - λ * scale) / sum(abs2, X)
    objective(coef) = loss(QuadraticLoss(), X * coef, y) + λ * scale * sum(abs, coef)

    for features in (X, sparse(X))
        fit = cdsolver(features, y, reg; intercept_strategy = NoIntercept(), save_history = true)
        @test fit.passes < 1000
        @test fit.intercept == 0.0
        @test only(fit.coef) ≈ expected
        @test fit.primals ≈ objective.(eachcol(fit.coefs))
        @test fit.primals[end] ≈ objective(fit.coef)
        @test all(iszero, fit.intercepts)

        warm = cdsolver(
            features, y, reg; intercept_strategy = NoIntercept(),
            coef_init = fit.coef, intercept_init = 12.0,
        )
        @test warm.passes == 1
        @test warm.intercept == 0.0
        @test warm.coef ≈ fit.coef
        @test warm.primals[end] ≈ objective(warm.coef)
    end

    fit = gdsolver(X, y, reg; intercept_strategy = NoIntercept())
    @test fit.passes < 1000
    @test fit.intercept == 0.0
    @test only(fit.coef) ≈ expected
    @test fit.primals[end] ≈ objective(fit.coef)
end

@testset "NoIntercept logistic fits agree with explicit scaling" begin
    X = reshape(collect(1.0:8.0), :, 1)
    y = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    scales = std(X; dims = 1, corrected = false)
    f = LogisticLoss()

    for solver in (cdsolver, irlssolver), features in (X, sparse(X))
        options = (lossfun = f, intercept_strategy = NoIntercept(), save_history = true)
        fit = solver(features, y; options...)
        reference = solver(features ./ scales, y; normalization = :none, options...)
        objective(coef) = loss(f, X * coef, y) + fit.λ * sum(abs, coef .* vec(scales))

        @test fit.relgaps[end] <= 1.0e-10
        @test fit.intercept == 0.0
        @test fit.coef .* vec(scales) ≈ reference.coef
        @test fit.primals ≈ objective.(eachcol(fit.coefs))
        @test fit.primals[end] ≈ objective(fit.coef)
        @test all(iszero, fit.intercepts)
    end
end

@testset "NoIntercept multinomial fits agree with explicit scaling" begin
    X = reshape(collect(1.0:6.0), :, 1)
    y = [1, 1, 2, 3, 3, 3]
    scales = std(X; dims = 1, corrected = false)
    f = MultinomialLogisticLoss(K = 3)

    for features in (X, sparse(X))
        options = (lossfun = f, intercept_strategy = NoIntercept(), save_history = true)
        fit = multinomial_cdsolver(features, y; options...)
        reference = multinomial_cdsolver(features ./ scales, y; normalization = :none, options...)
        objective(coef) = loss(f, X * coef, y) + fit.λ * sum(abs, coef .* vec(scales))

        @test fit.relgaps[end] <= 1.0e-10
        @test fit.intercept == zeros(2)
        @test fit.coef .* vec(scales) ≈ reference.coef
        @test fit.primals ≈ objective.(fit.coefs)
        @test fit.primals[end] ≈ objective(fit.coef)
        @test all(intercept -> all(iszero, intercept), fit.intercepts)
    end
end
