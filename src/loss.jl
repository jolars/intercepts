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

function dual(::LogisticLoss, θ::AbstractVector, y::AbstractVector)
    value = 0.0
    tol = 64 * eps(Float64)
    for u_raw in θ .+ y
        isfinite(u_raw) || throw(ArgumentError("logistic dual means must be finite"))
        if u_raw < -tol || u_raw > 1 + tol
            throw(DomainError(u_raw, "logistic dual means must lie in [0, 1]"))
        end
        u = clamp(u_raw, 0.0, 1.0)
        value -= iszero(u) ? 0.0 : u * log(u)
        value -= isone(u) ? 0.0 : (1 - u) * log1p(-u)
    end
    return value
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
    return if !ok
        throw(
            ArgumentError(
                "Response variable for LogisticLoss regression must be binary (0 or 1)",
            )
        )
    end
end

# PoissonLoss regression
function loss(::PoissonLoss, η::AbstractVector, y::AbstractVector)
    return sum(exp.(η) .- y .* η)
end

function dual(::PoissonLoss, θ::AbstractVector, y::AbstractVector)
    if any(!isfinite, θ)
        throw(ArgumentError("θ must be finite"))
    end

    value = 0.0
    tol = 64 * eps(Float64)
    for u_raw in θ .+ y
        if u_raw < -tol
            throw(DomainError(u_raw, "Poisson dual means must be nonnegative"))
        end
        u = max(u_raw, 0.0)
        value += iszero(u) ? 0.0 : u * (1 - log(u))
    end
    return value
end

# Centering a residual enforces the unpenalized-intercept constraint but can
# leave the conjugate domain. Blending from the intercept-only feasible point
# keeps that constraint while moving as far as possible toward the centered
# residual. Scaling this result toward zero later also preserves feasibility.
function _dual_point(
        f::QuadraticLoss,
        η::AbstractVector,
        y::AbstractVector,
        fit_intercept::Bool,
    )
    r = residual(f, η, y)
    if fit_intercept
        r .-= mean(r)
    end
    return r
end

function _scalar_dual_point(
        f::LossFunction,
        η::AbstractVector,
        y::AbstractVector,
        fit_intercept::Bool,
        upper::Real,
    )
    r = residual(f, η, y)
    fit_intercept || return r

    r .-= mean(r)
    response_mean = mean(y)
    α = 1.0

    for i in eachindex(r, y)
        target = y[i] + r[i]
        direction = target - response_mean
        if direction < 0
            α = min(α, response_mean / -direction)
        elseif direction > 0 && isfinite(upper)
            α = min(α, (upper - response_mean) / direction)
        end
    end

    α = clamp(α, 0.0, 1.0)
    if 0 < α < 1
        # Keep roundoff from crossing a domain boundary attained by the exact step.
        α *= 1 - sqrt(eps(Float64))
    end
    for i in eachindex(r, y)
        anchor = response_mean - y[i]
        r[i] = anchor + α * (r[i] - anchor)
    end
    return r
end

function _dual_point(
        f::LogisticLoss,
        η::AbstractVector,
        y::AbstractVector,
        fit_intercept::Bool,
    )
    return _scalar_dual_point(f, η, y, fit_intercept, 1.0)
end

function _dual_point(
        f::PoissonLoss,
        η::AbstractVector,
        y::AbstractVector,
        fit_intercept::Bool,
    )
    return _scalar_dual_point(f, η, y, fit_intercept, Inf)
end

function _dual_scale(gradient, λ::Real)
    gradient_norm = norm(gradient, Inf)
    iszero(gradient_norm) && return 1.0
    return max(1.0, gradient_norm / λ)
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
        throw(
            ArgumentError(
                "Response variable for Poisson regression must be integer-valued counts",
            )
        )
    end
    return if length(unique(y)) < 2
        throw(
            ArgumentError(
                "Response variable for Poisson regression must contain at least two distinct values",
            )
        )
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

function dual(
        f::MultinomialLogisticLoss,
        Θ::AbstractMatrix{<:Real},
        y::AbstractVector{<:Integer},
    )
    n, Km1 = size(Θ)
    K = f.K
    size(Θ, 2) == K - 1 || throw(DimensionMismatch("Θ must have K - 1 columns"))
    length(y) == n || throw(DimensionMismatch("Θ and y must have the same row count"))

    value = 0.0
    tol = 64 * eps(Float64)
    @inbounds for i in 1:n
        ref = 1.0
        for k in 1:Km1
            u_raw = Θ[i, k] + (y[i] == k ? 1.0 : 0.0)
            isfinite(u_raw) ||
                throw(ArgumentError("multinomial dual probabilities must be finite"))
            if u_raw < -tol
                throw(DomainError(u_raw, "multinomial dual rows must lie in the simplex"))
            end
            u = max(u_raw, 0.0)
            value -= iszero(u) ? 0.0 : u * log(u)
            ref -= u_raw
        end
        if ref < -tol
            throw(DomainError(ref, "multinomial dual rows must lie in the simplex"))
        end
        ref = max(ref, 0.0)
        value -= iszero(ref) ? 0.0 : ref * log(ref)
    end
    return value
end

function _dual_point(
        f::MultinomialLogisticLoss,
        η::AbstractMatrix{<:Real},
        y::AbstractVector{<:Integer},
        fit_intercept::Bool,
    )
    Θ = residual(f, η, y)
    fit_intercept || return Θ

    n, Km1 = size(Θ)
    K = f.K
    for k in 1:Km1
        @views Θ[:, k] .-= mean(Θ[:, k])
    end

    counts = zeros(Float64, K)
    for yi in y
        counts[yi] += 1
    end
    proportions = counts ./ n
    proportions[K] = 1 - sum(@view proportions[1:Km1])

    α = 1.0
    for i in 1:n
        target_ref = 1.0
        for k in 1:Km1
            target = Θ[i, k] + (y[i] == k ? 1.0 : 0.0)
            direction = target - proportions[k]
            if direction < 0
                α = min(α, proportions[k] / -direction)
            end
            target_ref -= target
        end
        ref_direction = target_ref - proportions[K]
        if ref_direction < 0
            α = min(α, proportions[K] / -ref_direction)
        end
    end

    α = clamp(α, 0.0, 1.0)
    if 0 < α < 1
        # Keep roundoff from crossing a simplex face attained by the exact step.
        α *= 1 - sqrt(eps(Float64))
    end
    for i in 1:n, k in 1:Km1
        anchor = proportions[k] - (y[i] == k ? 1.0 : 0.0)
        Θ[i, k] = anchor + α * (Θ[i, k] - anchor)
    end
    return Θ
end

function validateresponse(f::MultinomialLogisticLoss, y::AbstractVector{<:Integer})
    K = f.K
    if any(y .< 1) || any(y .> K)
        throw(
            ArgumentError(
                "Response for MultinomialLogisticLoss must be class indices in 1..$(K)",
            )
        )
    end
    return if length(unique(y)) < 2
        throw(ArgumentError("Response variable must contain at least two distinct classes"))
    end
end
