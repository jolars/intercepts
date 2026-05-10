using Test
using Intercepts
using Random
using GLM
using Statistics

@testset "Intercepts" begin
    @testset "Coordinate Descent" begin
        include("cd.jl")
    end
    @testset "Multinomial" begin
        include("multinomial.jl")
    end
end

