using Test
using Intercepts
using Random
using GLM
using Statistics

@testset "Intercepts" begin
    @testset "Public API documentation" begin
        undocumented = filter(names(Intercepts)) do name
            name != :Intercepts && !Base.Docs.hasdoc(Intercepts, name)
        end
        @test isempty(undocumented)
    end
    @testset "Coordinate Descent" begin
        include("cd.jl")
    end
    @testset "Multinomial" begin
        include("multinomial.jl")
    end
    @testset "IRLS" begin
        include("irls.jl")
    end
    @testset "Datasets" begin
        include("datasets.jl")
    end
end
