using Random
using LinearAlgebra
using SparseArrays
using GLM

"""
    generatedata(n, p; kwargs...)

Generate a synthetic design matrix, response, and true coefficients for the
Gaussian, logistic, Poisson, and multinomial experiments. Keyword arguments
control the response family, feature distribution, correlation, sparsity,
signal amplitude, and multinomial class probabilities.
"""
function generatedata(
    n::Int,
    p::Int;
    response::Symbol = :normal,
    μ0::Real = 0.5,
    x_density::Real = 0.1,
    x_type = :normal,
    means::Union{AbstractVector, Symbol, Nothing} = nothing,
    ρ::Real = 0,
    s::Int = 5,
    type::Symbol = :constant,
    amplitude::Real = 1,
    K::Int = 3,
    class_probs::Union{AbstractVector, Nothing} = nothing,
)
    s = min(s, p)
    β = zeros(p)

    ind = Int.(round.(range(1, p; length = s)))

    if type == :constant
        β[ind] .= 1
    elseif type == :rnorm
        β[ind] .= randn(s)
    elseif type == :uniform
        β[ind] .= rand(s) * 2 .- 1
    end

    β .*= amplitude

    if x_type == :normal
        x = randn(n, p)

        if means isa AbstractVector
            if length(means) != p
                throw(ArgumentError(
                    "Length of means vector must match number of features (p).",
                ))
            end
            x .+= means'
        end

        if means == :random
            ms = randn(p)
            x .+= ms'
        end

        if ρ != 0
            inds = 1:p
            Σ = Symmetric(ρ.^abs.(inds .- inds'))
            chol = cholesky(Σ)
            x = x * chol.L
        end
    elseif x_type == :binary
        x = Float64.(sprand(Bool, n, p, Float64(x_density)))

        if ρ != 0
            for j in 2:p
                for i in 1:n
                    if rand() < ρ
                        x[i, j] = x[i, j - 1]
                    end
                end
            end
        end
    else
        throw(ArgumentError("Unsupported x_type: $x_type"))
    end

    if response == :multinomial
        π = class_probs === nothing ? fill(1.0 / K, K) : Vector{Float64}(class_probs)
        if length(π) != K
            throw(ArgumentError("class_probs length must equal K (got $(length(π)) vs $K)"))
        end
        if !isapprox(sum(π), 1.0; atol = 1.0e-8)
            throw(ArgumentError("class_probs must sum to 1"))
        end

        # Per-class sparse signal: each class k=1..K-1 gets s active features
        # at evenly-spaced indices, scaled by amplitude. Class K is reference (β=0).
        Bcoef = zeros(p, K - 1)
        for k in 1:(K - 1)
            offset = ((k - 1) * s) % p
            inds_k = mod1.(ind .+ offset, p)
            Bcoef[inds_k, k] .= amplitude
        end

        η = x * Bcoef # n × (K-1)

        # Reference-class intercepts that target the marginal π.
        β0 = log.(π[1:(K - 1)] ./ π[K])
        η .+= β0'

        # Sample classes by softmax.
        y = Vector{Int}(undef, n)
        for i in 1:n
            denom = 1.0
            cum = zeros(K)
            cum[K] = 1.0
            for k in 1:(K - 1)
                cum[k] = exp(η[i, k])
                denom += cum[k]
            end
            for k in 1:K
                cum[k] /= denom
            end
            # cumulative for sampling
            csum = 0.0
            r = rand()
            yi = K
            for k in 1:K
                csum += cum[k]
                if r <= csum
                    yi = k
                    break
                end
            end
            y[i] = yi
        end
        return x, y
    end

    η = x * β

    # Create distribution object
    if response == :normal
        dist = Normal()
        link = canonicallink(dist)
    elseif response == :binomial
        dist = Bernoulli()
        link = canonicallink(dist)
    elseif response == :poisson
        dist = Poisson()
        link = canonicallink(dist)
    else
        throw(ArgumentError("Unsupported family: $response"))
    end

    β0 = GLM.linkfun.(link, μ0)
    η .+= β0
    μ = GLM.linkinv.(link, η)

    y = Vector{eltype(μ)}(undef, n)

    for i in 1:n
        ddist = if response == :normal
            Normal(μ[i], 1)
        elseif response == :binomial
            Bernoulli(μ[i])
        elseif response == :poisson
            Poisson(μ[i])
        end
        y[i] = rand(ddist)
    end

    return x, y
end
