using LinearAlgebra
using Statistics
using Random
using Arpack

function gdsolver(
  x::AbstractMatrix,
  y::AbstractVector,
  reg::Real=0.1;
  lossfun::LossFunction=Quadratic(),
  intercept_strategy::InterceptStrategy=GradientStrategy(),
  tol::Real=1e-10,
  maxit::Int=1000,
)
  n, p = size(x)

  intercept = 0.0

  coef = zeros(p)
  r = zeros(n)
  η = zeros(n)

  intercept = update_intercept(intercept_strategy, lossfun, intercept, η, y)

  η .+= intercept

  r = residual(lossfun, η, y)

  λmax = norm(x' * r, Inf)

  λ = reg * λmax

  if issparse(x)
    L = svds(x, nsv=1)[1].S[1]
  else
    L = opnorm(x)^2
  end

  L *= lossfun.lipschitz

  primals = Float64[]
  duals = Float64[]
  gaps = Float64[]

  times = Float64[]
  t0 = time()

  it = 0

  while it < maxit
    it += 1

    primal = loss(lossfun, η, y) + λ * norm(coef, 1)

    r = residual(lossfun, η, y)
    θ = r .- mean(r)
    dual_scale = max(1, norm(x' * θ, Inf) / λ)
    θ ./= dual_scale
    dua = dual(lossfun, θ, y)
    gap = primal - dua

    push!(primals, primal)
    push!(duals, dua)
    push!(times, time() - t0)

    rel_gap = gap / max(abs(primal), 1e-15)

    push!(gaps, gap)

    if rel_gap <= tol
      @info "Converged with gap: $rel_gap"
      break
    end

    r = residual(lossfun, η, y)

    old_intercept = intercept

    grad = x' * r
    coef = st.(coef - grad / L, λ / L)
    η = x * coef .+ old_intercept

    intercept = update_intercept(intercept_strategy, lossfun, old_intercept, η, y)

    η .+= intercept - old_intercept
  end

  return (intercept=intercept, coef=coef, primals=primals, duals=duals, gaps=gaps, time=times, passes=it)
end
