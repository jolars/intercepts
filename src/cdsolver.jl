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
  randomize::Bool=true,
  save_history::Bool=false,
)
  n, p = size(x)

  intercept = 0.0

  update_when = floor(size(x, 2) / update_freq)

  coef = zeros(p)
  r = zeros(n)
  η = zeros(n)

  intercept = linkfun(link, mean(y))
  η .+= intercept

  λmax = lambdamax(lossfun, x, y)
  λ = reg * λmax

  r = residual(lossfun, η, y)

  intercepts = Vector{Float64}(undef, 0)
  coefs = Vector{Vector{Float64}}(undef, 0)

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

    if save_history
      push!(intercepts, intercept)
      push!(coefs, copy(coef))
    end

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

    coef_old = copy(coef)
    intercept_old = intercept

    for (inner_it, j) in enumerate(ind)
      coef_j = coef[j]

      grad = dot(x[:, j], gradient(lossfun, η, y))
      hess = dot(x[:, j], x[:, j] .* hessian(lossfun, η, y))

      if hess == 0
        hess = 1e-10
      end

      coef[j] = st(coef_j - grad / hess, λ / hess)

      diff = coef_j - coef[j]
      if diff != 0
        η .-= diff * x[:, j]
      end

      if inner_it % update_when == 0
        intercept_prev = intercept
        intercept = update_intercept(intercept_strategy, lossfun, intercept_prev, η, y)
        η .+= intercept - intercept_prev
      end
    end

    old_primal = primal
    primal = loss(lossfun, η, y) + λ * norm(coef, 1)

    coef_diff = coef - coef_old
    intercept_diff = intercept - intercept_old

    alpha = 1.0

    while primal > old_primal && alpha > 1e-4
      alpha *= 0.1

      coef = coef_old + alpha * coef_diff
      intercept = intercept_old + alpha * intercept_diff

      η = x * coef .+ intercept
      primal = loss(lossfun, η, y) + λ * norm(coef, 1)
    end
  end

  return (
    intercept=intercept,
    coef=coef,
    primals=primals,
    duals=duals,
    gaps=gaps,
    time=times,
    passes=it,
    coefs=coefs,
    intercepts=intercepts,
  )
end
