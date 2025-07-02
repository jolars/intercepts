abstract type InterceptStrategy end

struct NoIntercept <: InterceptStrategy end
struct GradientStrategy <: InterceptStrategy end
struct NewtonStrategy <: InterceptStrategy end
struct ExactStrategy <: InterceptStrategy end

function update_intercept(::NoIntercept, f::LossFunction, intercept::Real, η::AbstractVector{<:Real}, y::AbstractVector{<:Real})
  0
end

function update_intercept(::GradientStrategy, f::LossFunction, intercept::Real, η::AbstractVector{<:Real}, y::AbstractVector{<:Real})
  return intercept - sum(gradient(f, η, y)) / (f.lipschitz * length(η))
end

function update_intercept(::NewtonStrategy, f::LossFunction, intercept::Real, η::AbstractVector{<:Real}, y::AbstractVector{<:Real})
  grad = sum(gradient(f, η, y))
  hess = sum(hessian(f, η, y))

  return intercept - grad / hess
end

# Iteratively update the intercept until convergence
function update_intercept(::ExactStrategy, f::LossFunction, intercept::Real, η::AbstractVector{<:Real}, y::AbstractVector{<:Real})
  η_copy = copy(η)

  max_it = 15

  for _ in 1:max_it
    grad = sum(gradient(f, η_copy, y))

    if abs(grad) < 1e-14
      return intercept
    end

    hess = sum(hessian(f, η_copy, y))

    new_intercept = intercept - grad / hess

    η_copy .+= new_intercept - intercept

    intercept = new_intercept

    abs_grad = abs(grad)
  end

  return intercept
end
