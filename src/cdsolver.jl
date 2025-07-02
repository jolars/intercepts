using LinearAlgebra
using Statistics
using Random

function cdsolver(
  x::AbstractMatrix,
  y::AbstractVector,
  reg::Real=0.1;
  lossfun::LossFunction=Quadratic(),
  intercept_strategy::InterceptStrategy=GradientStrategy(),
  update_freq::Int=1,
  tol::Real=1e-10,
  maxit::Int=1000,
  randomize::Bool=true
)
  n, p = size(x)

  intercept = 0.0

  update_when = floor(size(x, 2) / update_freq)

  coef = zeros(p)
  r = zeros(n)
  η = zeros(n)

  intercept = update_intercept(intercept_strategy, lossfun, intercept, η, y)

  η .+= intercept

  r = residual(lossfun, η, y)

  λmax = norm(x' * r, Inf)

  λ = reg * λmax

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

    ind = if randomize
      randperm(p)
    else
      1:p
    end

    for (inner_it, j) in enumerate(ind)
      coef_j = coef[j]

      grad = dot(x[:, j], gradient(lossfun, η, y))
      hess = dot(x[:, j], x[:, j] .* hessian(lossfun, η, y))

      coef[j] = st(coef_j - grad / hess, λ / hess)

      diff = coef_j - coef[j]
      if diff != 0
        η .-= diff * x[:, j]
      end

      if inner_it % update_when == 0
        intercept_old = intercept
        intercept = update_intercept(intercept_strategy, lossfun, intercept_old, η, y)
        η .+= intercept - intercept_old
      end
    end
  end

  return (intercept=intercept, coef=coef, primals=primals, duals=duals, gaps=gaps, time=times, passes=it)
end
