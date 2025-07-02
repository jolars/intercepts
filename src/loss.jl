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
  0.5 * norm(η)^2
end

function dual(::Quadratic, θ::AbstractVector, y::AbstractVector)
  0.5 * (norm(y)^2 - norm(θ .+ y)^2)
end

function link(::Quadratic, μ::AbstractVector)
  μ
end

function invlink(::Quadratic, η::AbstractVector)
  η
end

function weight(::Quadratic, η::AbstractVector)
  ones(length(η))
end

function gradient(::Quadratic, η::AbstractVector, y::AbstractVector)
  η - y
end

function hessian(::Quadratic, η::AbstractVector, y::AbstractVector)
  ones(length(η))
end

# Logistic regression

function loss(::Logistic, η::AbstractVector, y::AbstractVector)
  sum(log1p.(exp.(η)) .- η .* y)
end

function dual(f::Logistic, θ::AbstractVector, y::AbstractVector)
  η = link(f, θ .+ y)

  return loss(f, η, y) - dot(θ, η)
end

function link(::Logistic, μ::AbstractVector)
  logit.(μ)
end

function invlink(::Logistic, η::AbstractVector)
  sigmoid.(η)
end

function weight(::Logistic, η::AbstractVector)
  s = sigmoid.(η)
  s .* (1 .- s)
end

function gradient(::Logistic, η::AbstractVector, y::AbstractVector)
  sigmoid.(η) - y
end

function hessian(::Logistic, η::AbstractVector, ::AbstractVector)
  sigmoid.(η) .* (1 .- sigmoid.(η))
end

function residual(f::LossFunction, η::AbstractVector, y::AbstractVector)
  invlink(f, η) - y
end

function workingresponse(f::LossFunction, η::AbstractVector, y::AbstractVector, w::AbstractVector)
  η + (y - invlink(f, η)) ./ w
end
