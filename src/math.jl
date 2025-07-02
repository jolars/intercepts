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
