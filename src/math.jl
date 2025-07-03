using LinearAlgebra
using SparseArrays
using Statistics

function logit(x::Real)
  pr = clamp(x, 1e-15, 1 - 1e-15)
  log(pr) - log1p(-pr)
end

function sigmoid(x::Real)
  1 / (1 + exp(-x))
end

function st(u::Float64, λ::Float64)
  sign(u) * max(abs(u) - λ, 0.0)
end

function lambdamax(f::LossFunction, x::AbstractMatrix, y::AbstractVector)
  n = size(x, 1)
  intercept = link(f, mean(y))
  η = fill(intercept, n)
  r = residual(f, η, y)

  return norm(x' * r, Inf)
end
