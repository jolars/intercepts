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

# Numerically stable softplus log(1 + exp(x)). The naive log1p(exp(x)) overflows
# once exp(x) saturates (x ≳ 709), which the saturated-logistic and extreme-rate
# regimes studied here can reach; the branch below stays finite, reducing to
# x + log1p(exp(-x)) → x as x grows.
function log1pexp(x::Real)
    return x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))
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

function geomspace(start::Real, stop::Real, num::Int)
    if start <= 0 || stop <= 0 || num <= 0
        throw(ArgumentError("start, stop, and num must be positive"))
    end
    return exp.(range(log(start), log(stop), length = num))
end

