using LinearAlgebra
using Statistics

abstract type LossFunction end

@kwdef struct Quadratic <: LossFunction
    lipschitz::Float64 = 1.0
end

@kwdef struct Logistic <: LossFunction
    lipschitz::Float64 = 0.25
end

# Quadratic loss
function loss(::Quadratic, η::AbstractVector, ::AbstractVector)
    return 0.5 * norm(η)^2
end

function dual(::Quadratic, θ::AbstractVector, y::AbstractVector)
    return 0.5 * (norm(y)^2 - norm(θ .+ y)^2)
end

function link(::Quadratic, μ::Union{Real,AbstractVector})
    return μ
end

function invlink(::Quadratic, η::AbstractVector)
    return η
end

function weight(::Quadratic, η::AbstractVector)
    return ones(length(η))
end

function gradient(::Quadratic, η::AbstractVector, y::AbstractVector)
    return η - y
end

function hessian(::Quadratic, η::AbstractVector, y::AbstractVector)
    return ones(length(η))
end

function validateresponse(::Quadratic, y::AbstractVector) end

# Logistic regression
function loss(::Logistic, η::AbstractVector, y::AbstractVector)
    return sum(log1p.(exp.(η)) .- η .* y)
end

function dual(f::Logistic, θ::AbstractVector, y::AbstractVector)
    η = link(f, θ .+ y)

    return loss(f, η, y) - dot(θ, η)
end

function link(::Logistic, μ::Real)
    return logit(μ)
end

function link(::Logistic, μ::AbstractVector)
    return logit.(μ)
end

function invlink(::Logistic, η::AbstractVector)
    return sigmoid.(η)
end

function weight(::Logistic, η::AbstractVector)
    s = sigmoid.(η)
    return s .* (1 .- s)
end

function gradient(::Logistic, η::AbstractVector, y::AbstractVector)
    return sigmoid.(η) - y
end

function hessian(::Logistic, η::AbstractVector, ::AbstractVector)
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

function validateresponse(::Logistic, y::AbstractVector)
    ok = unique(y) ⊆ (0.0, 1.0)
    if !ok
        throw(
            ArgumentError(
                "Response variable for logistic regression must be binary (0 or 1)",
            ),
        )
    end
end
