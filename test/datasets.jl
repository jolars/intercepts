using SparseArrays

@testset "load_libsvm" begin
    @testset "1-based parsing with padding" begin
        # Three samples; the third has no features, and feature column 4 is
        # never observed --- both must survive because the dimensions are fixed.
        lines = ["+1 1:0.5 3:1.0", "-1 2:2.0", "+1"]
        path = tempname()
        write(path, join(lines, "\n") * "\n")

        X, y = load_libsvm(path; n_samples = 3, n_features = 4)
        @test size(X) == (3, 4)
        @test y == [1.0, -1.0, 1.0]
        @test X[1, 1] == 0.5
        @test X[1, 3] == 1.0
        @test X[2, 2] == 2.0
        @test nnz(X) == 3
        @test all(iszero, X[3, :])

        Xd, yd = load_libsvm(path; n_samples = 3, n_features = 4, dense = true)
        @test Xd isa Matrix{Float64}
        @test Xd == Matrix(X)
        @test yd == y

        rm(path)
    end

    @testset "0-based indices are shifted" begin
        # A leading 0 index switches the file to 0-based; columns shift by one.
        path = tempname()
        write(path, "0 0:1.0 2:3.0\n")

        X, y = load_libsvm(path; n_samples = 1, n_features = 3)
        @test size(X) == (1, 3)
        @test y == [0.0]
        @test X[1, 1] == 1.0
        @test X[1, 3] == 3.0

        rm(path)
    end
end
