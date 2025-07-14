using Test
using Intercepts
using Random
using GLM
using Statistics
using Aqua

@testset "Intercepts" begin
    @testset "Coordinate Descent" begin
        include("cd.jl")
    end

    @testset "Aqua" begin
        Aqua.test_all(Intercepts)
    end
end

