using Test
using Intercepts
using Random
using Statistics

@testset "Gaussian design covariance" begin
    for ρ in (-0.6, 0.6)
        Random.seed!(2026)
        X, _ = generatedata(40_000, 4; ρ = ρ)
        target = [ρ^abs(j - k) for j in 1:4, k in 1:4]

        # Check the generated distribution, including unit marginal variances.
        @test maximum(abs.(cov(X) .- target)) < 0.04
    end
end
