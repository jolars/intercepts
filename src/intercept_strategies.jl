using Statistics

abstract type InterceptStrategy end

struct NoIntercept <: InterceptStrategy end
struct GradientStrategy <: InterceptStrategy end
struct NewtonStrategy <: InterceptStrategy end
struct ConvergenceStrategy <: InterceptStrategy end

@kwdef struct DampedNewtonStrategy <: InterceptStrategy
    armijo_c::Float64 = 1.0e-4
    backtrack::Float64 = 0.5
    max_backtracks::Int = 20
end

function update_intercept(
    ::NoIntercept,
    f::LossFunction,
    intercept::Real,
    η::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
)
    0
end

function update_intercept(
    ::GradientStrategy,
    f::LossFunction,
    intercept::Real,
    η::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
)
    return intercept - sum(gradient(f, η, y)) / (f.lipschitz * length(η))
end

function update_intercept(
    ::NewtonStrategy,
    f::LossFunction,
    intercept::Real,
    η::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
)
    grad = sum(gradient(f, η, y))
    hess = sum(hessian(f, η, y))

    return intercept - grad / hess
end

# Newton step with Armijo backtracking line search
function update_intercept(
    s::DampedNewtonStrategy,
    f::LossFunction,
    intercept::Real,
    η::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
)
    grad = sum(gradient(f, η, y))
    hess = sum(hessian(f, η, y))

    if hess <= 0 || !isfinite(hess) || grad == 0
        return intercept
    end

    direction = -grad / hess
    f0 = loss(f, η, y)
    armijo_slope = s.armijo_c * grad * direction

    α = 1.0
    η_trial = similar(η)
    for _ = 0:s.max_backtracks
        @. η_trial = η + α * direction
        f_trial = loss(f, η_trial, y)
        if f_trial <= f0 + α * armijo_slope
            return intercept + α * direction
        end
        α *= s.backtrack
    end

    return intercept + α * direction
end

# Iteratively update the intercept until convergence
function update_intercept(
    ::ConvergenceStrategy,
    f::LossFunction,
    intercept::Real,
    η::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
)
    η_copy = copy(η)
    n = length(η)

    max_it = 20

    for _ = 1:max_it
        grad = sum(gradient(f, η_copy, y))

        if abs(grad) < 1e-10
            break
        end

        hess = sum(hessian(f, η_copy, y))

        if abs(hess) < 1e-10
            # If Hessian is small, use standard gradient descent step
            new_intercept = intercept - grad / (f.lipschitz * n)
        else
            new_intercept = intercept - grad / hess
        end

        η_copy .+= new_intercept - intercept

        intercept = new_intercept
    end

    return intercept
end
