using LinearAlgebra
using Statistics

"""Abstract interface implemented by the supported GLM loss functions."""
abstract type LossFunction end

"""Squared-error loss with identity link."""
@kwdef struct QuadraticLoss <: LossFunction
    lipschitz::Float64 = 1.0
end

"""Bernoulli negative log-likelihood with logistic link."""
@kwdef struct LogisticLoss <: LossFunction
    lipschitz::Float64 = 0.25
end

"""Poisson negative log-likelihood with log link."""
@kwdef struct PoissonLoss <: LossFunction
    lipschitz::Float64 = Inf
end

# QuadraticLoss loss
"""Evaluate the negative log-likelihood at linear predictor `η` and response `y`."""
function loss(::QuadraticLoss, η::AbstractVector, y::AbstractVector)
    return 0.5 * sum(abs2, η .- y)
end

"""Evaluate the Fenchel dual objective for a loss and dual variable `θ`."""
function dual(::QuadraticLoss, θ::AbstractVector, y::AbstractVector)
    return 0.5 * (norm(y)^2 - norm(θ .+ y)^2)
end

"""Map a mean response `μ` to its linear-predictor scale."""
function link(::QuadraticLoss, μ::Union{Real, AbstractVector})
    return μ
end

"""Map a linear predictor `η` to the mean-response scale."""
function invlink(::QuadraticLoss, η::AbstractVector)
    return η
end

"""Return the pointwise curvature weights at `η`."""
function weight(::QuadraticLoss, η::AbstractVector)
    return ones(length(η))
end

"""Return the pointwise loss gradient with respect to `η`."""
function gradient(::QuadraticLoss, η::AbstractVector, y::AbstractVector)
    return η - y
end

"""Return the pointwise loss Hessian with respect to `η`."""
function hessian(::QuadraticLoss, η::AbstractVector, y::AbstractVector)
    return ones(length(η))
end

function validateresponse(::QuadraticLoss, y::AbstractVector) end

# LogisticLoss regression
function loss(::LogisticLoss, η::AbstractVector, y::AbstractVector)
    # log1pexp keeps the loss finite under saturation (η ≳ 709), where the naive
    # log1p(exp(η)) would overflow to Inf.
    return sum(log1pexp.(η) .- η .* y)
end

function dual(f::LogisticLoss, θ::AbstractVector, y::AbstractVector)
    η = link(f, θ .+ y)

    return loss(f, η, y) - dot(θ, η)
end

function link(::LogisticLoss, μ::Real)
    return logit(μ)
end

function link(::LogisticLoss, μ::AbstractVector)
    return logit.(μ)
end

function invlink(::LogisticLoss, η::AbstractVector)
    return sigmoid.(η)
end

function weight(::LogisticLoss, η::AbstractVector)
    s = sigmoid.(η)
    return s .* (1 .- s)
end

function gradient(::LogisticLoss, η::AbstractVector, y::AbstractVector)
    return sigmoid.(η) - y
end

function hessian(::LogisticLoss, η::AbstractVector, ::AbstractVector)
    return sigmoid.(η) .* (1 .- sigmoid.(η))
end

"""Return `invlink(f, η) - y`, the response-scale residual."""
function residual(f::LossFunction, η::AbstractVector, y::AbstractVector)
    return invlink(f, η) - y
end

"""Construct the IRLS working response from `η`, `y`, and weights `w`."""
function workingresponse(
    f::LossFunction,
    η::AbstractVector,
    y::AbstractVector,
    w::AbstractVector,
)
    return η + (y - invlink(f, η)) ./ w
end

function validateresponse(::LogisticLoss, y::AbstractVector)
    ok = unique(y) ⊆ (0.0, 1.0)
    if !ok
        throw(ArgumentError(
            "Response variable for LogisticLoss regression must be binary (0 or 1)",
        ))
    end
end

# PoissonLoss regression
function loss(::PoissonLoss, η::AbstractVector, y::AbstractVector)
    return sum(exp.(η) .- y .* η)
end

function dual(::PoissonLoss, θ::AbstractVector, y::AbstractVector)
    # check if eta is finite
    if any(!isfinite, θ)
        throw(ArgumentError("θ must be finite"))
    end

    e = θ .+ y
    e = max.(e, 1.0e-15) # clamp to avoid log(0)
    return sum(e .* (1 .- log.(e)))
end

function link(::PoissonLoss, μ::Real)
    return log(max(μ, 1.0e-15))
end

function link(::PoissonLoss, μ::AbstractVector)
    return log.(max.(μ, 1.0e-15))
end

function invlink(::PoissonLoss, η::AbstractVector)
    return exp.(η)
end

function weight(::PoissonLoss, η::AbstractVector)
    return exp.(η)
end

function gradient(::PoissonLoss, η::AbstractVector, y::AbstractVector)
    return exp.(η) .- y
end

function hessian(::PoissonLoss, η::AbstractVector, ::AbstractVector)
    return exp.(η)
end

function validateresponse(::PoissonLoss, y::AbstractVector)
    if any(y .< 0)
        throw(ArgumentError("Response must be non-negative"))
    end
    if !all(isinteger.(y))
        throw(ArgumentError(
            "Response variable for Poisson regression must be integer-valued counts",
        ))
    end
    if length(unique(y)) < 2
        throw(ArgumentError(
            "Response variable for Poisson regression must contain at least two distinct values",
        ))
    end
end

# MultinomialLogisticLoss: K-class softmax with reference class K (η_iK = 0).
# η is stored as Matrix(n, K-1) for the K-1 free classes.
# y is a Vector{Int} of class labels in 1..K.
"""Multinomial logistic loss with class `K` as the reference class."""
@kwdef struct MultinomialLogisticLoss <: LossFunction
    K::Int
    lipschitz::Float64 = 0.5
end

"""
    softmax_probs(η)

Return the `n × K` matrix of softmax probabilities for an `n × (K - 1)` matrix
of linear predictors. The omitted final column is the reference class, whose
linear predictor is zero.
"""
function softmax_probs(η::AbstractMatrix{<:Real})
    n, Km1 = size(η)
    K = Km1 + 1
    p = Matrix{Float64}(undef, n, K)
    @inbounds for i in 1:n
        m = 0.0
        for k in 1:Km1
            ηik = η[i, k]
            if ηik > m
                m = ηik
            end
        end
        s = exp(-m)
        for k in 1:Km1
            p[i, k] = exp(η[i, k] - m)
            s += p[i, k]
        end
        p[i, K] = exp(-m)
        for k in 1:K
            p[i, k] /= s
        end
    end
    return p
end

function loss(
    f::MultinomialLogisticLoss,
    η::AbstractMatrix{<:Real},
    y::AbstractVector{<:Integer},
)
    n, Km1 = size(η)
    K = f.K
    L = 0.0
    @inbounds for i in 1:n
        m = 0.0
        for k in 1:Km1
            ηik = η[i, k]
            if ηik > m
                m = ηik
            end
        end
        s = exp(-m)
        for k in 1:Km1
            s += exp(η[i, k] - m)
        end
        lse = m + log(s)
        ηy = (y[i] == K) ? 0.0 : η[i, y[i]]
        L += lse - ηy
    end
    return L
end

function gradient(
    f::MultinomialLogisticLoss,
    η::AbstractMatrix{<:Real},
    y::AbstractVector{<:Integer},
)
    p = softmax_probs(η)
    n, Km1 = size(η)
    G = Matrix{Float64}(undef, n, Km1)
    @inbounds for i in 1:n
        for k in 1:Km1
            G[i, k] = p[i, k] - (y[i] == k ? 1.0 : 0.0)
        end
    end
    return G
end

# Per-observation Hessian blocks: H[i, k, l] = p[i, k] (𝟙{k=l} - p[i, l]).
# Returns Array(n, K-1, K-1).
function hessian(
    f::MultinomialLogisticLoss,
    η::AbstractMatrix{<:Real},
    ::AbstractVector{<:Integer},
)
    p = softmax_probs(η)
    n, Km1 = size(η)
    H = Array{Float64, 3}(undef, n, Km1, Km1)
    @inbounds for i in 1:n
        for k in 1:Km1
            pik = p[i, k]
            for l in 1:Km1
                H[i, k, l] = (k == l) ? pik * (1 - pik) : -pik * p[i, l]
            end
        end
    end
    return H
end

function invlink(f::MultinomialLogisticLoss, η::AbstractMatrix{<:Real})
    return softmax_probs(η)
end

# Returns the n × K softmax probability matrix (analog of LogisticLoss weight).
function weight(f::MultinomialLogisticLoss, η::AbstractMatrix{<:Real})
    return softmax_probs(η)
end

function residual(
    f::MultinomialLogisticLoss,
    η::AbstractMatrix{<:Real},
    y::AbstractVector{<:Integer},
)
    return gradient(f, η, y)
end

# Inverse softmax link: recover the K-1 free-class linear predictors from a
# probability matrix P (n × K, rows summing to one) with reference class K
# pinned at η_iK = 0, so η_ik = log(P_ik) - log(P_iK).
function link(f::MultinomialLogisticLoss, P::AbstractMatrix{<:Real})
    n, K = size(P)
    η = Matrix{Float64}(undef, n, K - 1)
    @inbounds for i in 1:n
        logref = log(P[i, K])
        for k in 1:(K - 1)
            η[i, k] = log(P[i, k]) - logref
        end
    end
    return η
end

# Fenchel dual of the multinomial logistic loss at a dual point Θ (n × K-1).
# Θ is a dual-feasible variable (the intercept-centered, ℓ∞-rescaled residual
# built by the solver). The conjugate is evaluated by recovering its maximizing
# η: the implied class probabilities are u_ik = Θ_ik + 𝟙{y_i = k} for the free
# classes and u_iK = 1 - Σ_k u_ik for the reference, so the dual value is
# loss(η⋆) - ⟨Θ, η⋆⟩ with η⋆ = link(u). For the scaled residual u is a convex
# combination of the softmax row and a simplex vertex and so stays in the
# simplex; the small floor only guards the logarithm against a boundary point
# far from the optimum.
function dual(
    f::MultinomialLogisticLoss,
    Θ::AbstractMatrix{<:Real},
    y::AbstractVector{<:Integer},
)
    n, Km1 = size(Θ)
    K = f.K
    U = Matrix{Float64}(undef, n, K)
    @inbounds for i in 1:n
        ref = 1.0
        for k in 1:Km1
            u = Θ[i, k] + (y[i] == k ? 1.0 : 0.0)
            U[i, k] = u
            ref -= u
        end
        U[i, K] = ref
    end
    U .= max.(U, 1.0e-12)
    η = link(f, U)
    return loss(f, η, y) - sum(Θ .* η)
end

function validateresponse(f::MultinomialLogisticLoss, y::AbstractVector{<:Integer})
    K = f.K
    if any(y .< 1) || any(y .> K)
        throw(ArgumentError(
            "Response for MultinomialLogisticLoss must be class indices in 1..$(K)",
        ))
    end
    if length(unique(y)) < 2
        throw(ArgumentError("Response variable must contain at least two distinct classes"))
    end
end
