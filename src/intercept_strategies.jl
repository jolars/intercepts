using Statistics

abstract type InterceptStrategy end

struct NoIntercept <: InterceptStrategy end
struct GradientStrategy <: InterceptStrategy end
struct NewtonStrategy <: InterceptStrategy end
struct ExactStrategy <: InterceptStrategy end

"""
    DampedNewtonStrategy(; armijo_c = 1e-4, backtrack = 0.5, max_backtracks = 20)

Deprecated alias for [`NewtonStrategy`](@ref). The Newton intercept update
now performs its own Armijo line search, so the historically separate
damped variant produces the same iterates as the bare strategy. The
keyword arguments are accepted for backward compatibility but ignored;
`update_intercept` delegates to `NewtonStrategy()` in both the scalar and
vector dispatch.
"""
@kwdef struct DampedNewtonStrategy <: InterceptStrategy
    armijo_c::Float64 = 1.0e-4
    backtrack::Float64 = 0.5
    max_backtracks::Int = 20
end

@kwdef struct BacktrackingGradientStrategy <: InterceptStrategy
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

    if hess <= 0 || !isfinite(hess) || grad == 0
        return intercept
    end

    direction = -grad / hess
    f0 = loss(f, η, y)

    armijo_c = 1.0e-4
    armijo_shrink = 0.5
    armijo_max_backtracks = 20
    armijo_slope = armijo_c * grad * direction
    fp_tol = 4 * eps(Float64) * (1.0 + abs(f0))

    α = 1.0
    η_trial = similar(η)
    for _ = 0:armijo_max_backtracks
        @. η_trial = η + α * direction
        f_trial = loss(f, η_trial, y)
        if f_trial <= f0 + α * armijo_slope + fp_tol
            return intercept + α * direction
        end
        α *= armijo_shrink
    end

    return intercept
end

function update_intercept(
    ::DampedNewtonStrategy,
    f::LossFunction,
    intercept::Real,
    η::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
)
    return update_intercept(NewtonStrategy(), f, intercept, η, y)
end

# Gradient step (with global Lipschitz step size) protected by Armijo backtracking
function update_intercept(
    s::BacktrackingGradientStrategy,
    f::LossFunction,
    intercept::Real,
    η::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
)
    n = length(η)
    grad = sum(gradient(f, η, y))

    if grad == 0
        return intercept
    end

    direction = -grad / (f.lipschitz * n)
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
    ::ExactStrategy,
    f::LossFunction,
    intercept::Real,
    η::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
)
    new_intercept, _ = update_intercept_with_count(
        ExactStrategy(),
        f,
        intercept,
        η,
        y,
    )
    return new_intercept
end

function update_intercept_with_count(
    ::ExactStrategy,
    f::LossFunction,
    intercept::Real,
    η::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
)
    η_copy = copy(η)
    n = length(η)

    max_it = 20
    k0 = 0

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
        k0 += 1
    end

    return intercept, k0
end

# Generic fallback: any one-step strategy counts as a single inner step.
function update_intercept_with_count(
    s::InterceptStrategy,
    f::LossFunction,
    intercept,
    η,
    y,
)
    return update_intercept(s, f, intercept, η, y), 1
end

function update_intercept_with_count(
    ::NoIntercept,
    f::LossFunction,
    intercept,
    η,
    y,
)
    return update_intercept(NoIntercept(), f, intercept, η, y), 0
end


# =============================================================================
# Vector-intercept dispatch (multinomial logistic, reference-class form).
# η is Matrix(n, K-1); intercept is Vector(K-1); y is Vector{Int} (class labels).
# =============================================================================

# Sum gradient across observations → length K-1.
function _intercept_grad(f::LossFunction, η::AbstractMatrix, y::AbstractVector{<:Integer})
    G = gradient(f, η, y)
    return vec(sum(G; dims = 1))
end

# Sum per-observation Hessian blocks → (K-1) × (K-1).
function _intercept_hess(f::LossFunction, η::AbstractMatrix, y::AbstractVector{<:Integer})
    H = hessian(f, η, y)
    return dropdims(sum(H; dims = 1); dims = 1)
end

function update_intercept(
    ::NoIntercept,
    f::LossFunction,
    intercept::AbstractVector{<:Real},
    η::AbstractMatrix{<:Real},
    y::AbstractVector{<:Integer},
)
    return zeros(length(intercept))
end

function update_intercept(
    ::GradientStrategy,
    f::LossFunction,
    intercept::AbstractVector{<:Real},
    η::AbstractMatrix{<:Real},
    y::AbstractVector{<:Integer},
)
    n = size(η, 1)
    g = _intercept_grad(f, η, y)
    return intercept .- g ./ (f.lipschitz * n)
end

function update_intercept(
    ::NewtonStrategy,
    f::LossFunction,
    intercept::AbstractVector{<:Real},
    η::AbstractMatrix{<:Real},
    y::AbstractVector{<:Integer},
)
    n = size(η, 1)
    g = _intercept_grad(f, η, y)
    H = _intercept_hess(f, η, y)
    local δ
    try
        δ = -(H \ g)
    catch err
        if err isa LinearAlgebra.SingularException
            δ = -g ./ (f.lipschitz * n)
        else
            rethrow()
        end
    end
    if any(!isfinite, δ)
        δ = -g ./ (f.lipschitz * n)
    end
    if all(==(0), δ)
        return copy(intercept)
    end

    f0 = loss(f, η, y)

    armijo_c = 1.0e-4
    armijo_shrink = 0.5
    armijo_max_backtracks = 20
    armijo_slope = armijo_c * dot(g, δ)
    fp_tol = 4 * eps(Float64) * (1.0 + abs(f0))

    α = 1.0
    η_trial = similar(η)
    for _ = 0:armijo_max_backtracks
        @. η_trial = η + α * δ'
        f_trial = loss(f, η_trial, y)
        if f_trial <= f0 + α * armijo_slope + fp_tol
            return intercept .+ α .* δ
        end
        α *= armijo_shrink
    end

    return copy(intercept)
end

function update_intercept(
    ::DampedNewtonStrategy,
    f::LossFunction,
    intercept::AbstractVector{<:Real},
    η::AbstractMatrix{<:Real},
    y::AbstractVector{<:Integer},
)
    return update_intercept(NewtonStrategy(), f, intercept, η, y)
end

function update_intercept(
    s::BacktrackingGradientStrategy,
    f::LossFunction,
    intercept::AbstractVector{<:Real},
    η::AbstractMatrix{<:Real},
    y::AbstractVector{<:Integer},
)
    n = size(η, 1)
    g = _intercept_grad(f, η, y)

    if all(==(0), g)
        return intercept
    end

    δ = -g ./ (f.lipschitz * n)
    f0 = loss(f, η, y)
    armijo_slope = s.armijo_c * dot(g, δ)

    α = 1.0
    η_trial = similar(η)
    for _ = 0:s.max_backtracks
        @. η_trial = η + α * δ'
        f_trial = loss(f, η_trial, y)
        if f_trial <= f0 + α * armijo_slope
            return intercept .+ α .* δ
        end
        α *= s.backtrack
    end

    return intercept .+ α .* δ
end

function update_intercept(
    ::ExactStrategy,
    f::LossFunction,
    intercept::AbstractVector{<:Real},
    η::AbstractMatrix{<:Real},
    y::AbstractVector{<:Integer},
)
    new_intercept, _ = update_intercept_with_count(
        ExactStrategy(),
        f,
        intercept,
        η,
        y,
    )
    return new_intercept
end

function update_intercept_with_count(
    ::ExactStrategy,
    f::LossFunction,
    intercept::AbstractVector{<:Real},
    η::AbstractMatrix{<:Real},
    y::AbstractVector{<:Integer},
)
    n = size(η, 1)
    η_copy = copy(η)
    intercept = copy(intercept)

    max_it = 20
    k0 = 0
    for _ = 1:max_it
        g = _intercept_grad(f, η_copy, y)

        if maximum(abs, g) < 1e-10
            break
        end

        H = _intercept_hess(f, η_copy, y)

        local δ
        try
            δ = -(H \ g)
        catch
            δ = -g ./ (f.lipschitz * n)
        end
        if any(!isfinite, δ)
            δ = -g ./ (f.lipschitz * n)
        end

        η_copy .+= δ'  # broadcast across rows
        intercept .+= δ
        k0 += 1
    end

    return intercept, k0
end
