using LinearAlgebra
using Statistics

abstract type LossFunction end

@kwdef struct QuadraticLoss <: LossFunction
    lipschitz::Float64 = 1.0
end

@kwdef struct LogisticLoss <: LossFunction
    lipschitz::Float64 = 0.25
end

@kwdef struct PoissonLoss <: LossFunction
    lipschitz::Float64 = Inf
end

# QuadraticLoss loss
function loss(::QuadraticLoss, η::AbstractVector, ::AbstractVector)
    return 0.5 * norm(η)^2
end

function dual(::QuadraticLoss, θ::AbstractVector, y::AbstractVector)
    return 0.5 * (norm(y)^2 - norm(θ .+ y)^2)
end

function link(::QuadraticLoss, μ::Union{Real,AbstractVector})
    return μ
end

function invlink(::QuadraticLoss, η::AbstractVector)
    return η
end

function weight(::QuadraticLoss, η::AbstractVector)
    return ones(length(η))
end

function gradient(::QuadraticLoss, η::AbstractVector, y::AbstractVector)
    return η - y
end

function hessian(::QuadraticLoss, η::AbstractVector, y::AbstractVector)
    return ones(length(η))
end

function validateresponse(::QuadraticLoss, y::AbstractVector) end

# LogisticLoss regression
function loss(::LogisticLoss, η::AbstractVector, y::AbstractVector)
    return sum(log1p.(exp.(η)) .- η .* y)
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

function residual(f::LossFunction, η::AbstractVector, y::AbstractVector)
    return invlink(f, η) - y
end

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
        throw(
            ArgumentError(
                "Response variable for LogisticLoss regression must be binary (0 or 1)",
            ),
        )
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
    e = max.(e, 1e-15)  # clamp to avoid log(0)
    return sum(e .* (1 .- log.(e)))
end

function link(::PoissonLoss, μ::Real)
    return log(max(μ, 1e-15))
end

function link(::PoissonLoss, μ::AbstractVector)
    return log.(max.(μ, 1e-15))
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
        throw(ArgumentError("Response variable for Poisson regression must be integer-valued counts"))
    end
    if length(unique(y)) < 2
        throw(ArgumentError("Response variable for Poisson regression must contain at least two distinct values"))
    end
end
