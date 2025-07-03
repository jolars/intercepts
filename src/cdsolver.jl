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
  normalization::Symbol=:standardize,
  save_history::Bool=false,
)
  n, p = size(x)

  y = Float64.(y)

  validateresponse(lossfun, y)

  x, x_centers, x_scales = normalizefeatures(x, normalization)

  x_sparse_offset = x_centers ./ x_scales

  sparse_norm = issparse(x) && normalization != :none

  fit_intercept = !(intercept_strategy isa NoIntercept)

  intercept = 0.0

  update_when = floor(size(x, 2) / update_freq)

  coef = zeros(p)
  r = zeros(n)
  η = zeros(n)

  intercept = link(lossfun, mean(y))
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
    θ = r

    if fit_intercept
      θ .-= mean(r)
    end

    grad = x' * θ

    if sparse_norm
      θsum = sum(θ)
      for j in 1:p
        grad[j] -= x_sparse_offset[j] * θsum
      end
    end

    dual_scale = max(1, norm(grad, Inf) / λ)

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

      η_grad = gradient(lossfun, η, y)
      η_hess = hessian(lossfun, η, y)

      grad = dot(x[:, j], η_grad)
      hess = dot(x[:, j], x[:, j] .* η_hess)

      if sparse_norm
        grad -= x_sparse_offset[j] * sum(η_grad)
        hess -= 2 * x_sparse_offset[j] * dot(η_hess, x[:, j]) -
                x_sparse_offset[j]^2 * sum(η_hess)
      end

      if hess == 0
        hess = 1e-10
      end

      coef[j] = st(coef_j - grad / hess, λ / hess)

      diff = coef_j - coef[j]
      if diff != 0
        η .-= diff * x[:, j]
        if sparse_norm
          η .+= diff * x_sparse_offset[j]
        end
      end

      if inner_it % update_when == 0 && fit_intercept
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
      if sparse_norm
        η .-= dot(x_sparse_offset, coef)
      end

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
