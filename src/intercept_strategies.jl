using Statistics

"""Abstract type for rules that update an unpenalized model intercept."""
abstract type InterceptStrategy end

"""Disable intercept updates and fit a model with a zero intercept."""
struct NoIntercept <: InterceptStrategy end

"""Take one intercept gradient step using the loss Lipschitz bound."""
struct GradientStrategy <: InterceptStrategy end

"""Take one Armijo-guarded Newton step for the intercept."""
struct NewtonStrategy <: InterceptStrategy end

"""
    ExactStrategy(; gradient_tol=1e-10, maxit=50)

Minimize the conditional intercept problem with safeguarded Newton iterations.
The solve raises an error rather than silently returning an iterate that has not
reached `gradient_tol` within `maxit` iterations, except for a fivefold numerical
margin that prevents saturated floating-point predictors from stalling at the
tolerance boundary.
"""
@kwdef struct ExactStrategy <: InterceptStrategy
    gradient_tol::Float64 = 1.0e-10
    maxit::Int = 50
    armijo_c::Float64 = 1.0e-4
    backtrack::Float64 = 0.5
    max_backtracks::Int = 50
end

"""
    UnguardedNewtonStrategy()

Newton intercept update with the Armijo backtracking guard removed: always
takes the full step `δ = -∂₀F / H₀₀`. Used by the cold-start diagnostics
in @fig-cold-start and @fig-cold-start-poisson to isolate the contribution
of the safeguard inside [`NewtonStrategy`](@ref); not recommended for
production use.
"""
struct UnguardedNewtonStrategy <: InterceptStrategy end

"""Take a Lipschitz gradient step with configurable Armijo backtracking."""
@kwdef struct BacktrackingGradientStrategy <: InterceptStrategy
    armijo_c::Float64 = 1.0e-4
    backtrack::Float64 = 0.5
    max_backtracks::Int = 20
end

"""
    update_intercept(strategy, f, intercept, η, y)

Return the next intercept for `f` at linear predictor `η` and response `y`,
using the selected `strategy`. For multinomial logistic loss, `η` is an
`n × (K - 1)` matrix and the intercept is a length `K - 1` vector.
"""
function update_intercept(
        ::NoIntercept,
        f::LossFunction,
        intercept::Real,
        η::AbstractVector{<:Real},
        y::AbstractVector{<:Real},
    )
    return 0
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
    for _ in 0:armijo_max_backtracks
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
        ::UnguardedNewtonStrategy,
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

    return intercept - grad / hess
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
    for _ in 0:s.max_backtracks
        @. η_trial = η + α * direction
        f_trial = loss(f, η_trial, y)
        if f_trial <= f0 + α * armijo_slope
            return intercept + α * direction
        end
        α *= s.backtrack
    end

    return intercept + α * direction
end

# Safeguarded conditional minimization of the intercept.
function update_intercept(
        strategy::ExactStrategy,
        f::LossFunction,
        intercept::Real,
        η::AbstractVector{<:Real},
        y::AbstractVector{<:Real},
    )
    new_intercept, _ = update_intercept_with_count(strategy, f, intercept, η, y)
    return new_intercept
end

function update_intercept_with_count(
        strategy::ExactStrategy,
        f::LossFunction,
        intercept::Real,
        η::AbstractVector{<:Real},
        y::AbstractVector{<:Real},
    )
    η_copy = copy(η)
    n = length(η)

    k0 = 0

    for _ in 1:strategy.maxit
        grad = sum(gradient(f, η_copy, y))

        if abs(grad) / n <= strategy.gradient_tol
            return intercept, k0
        end

        hess = sum(hessian(f, η_copy, y))

        direction = if hess > 0 && isfinite(hess)
            -grad / hess
        elseif isfinite(f.lipschitz) && f.lipschitz > 0
            -grad / (f.lipschitz * n)
        else
            -grad / n
        end

        f0 = loss(f, η_copy, y)
        slope = strategy.armijo_c * grad * direction
        fp_tol = 4 * eps(Float64) * (1.0 + abs(f0))
        α = 1.0
        accepted = false
        for _ in 0:strategy.max_backtracks
            f_trial = loss(f, η_copy .+ α * direction, y)
            if f_trial <= f0 + α * slope + fp_tol
                accepted = true
                break
            end
            α *= strategy.backtrack
        end
        accepted || error("ExactStrategy line search failed to find a descent step")

        step = α * direction
        η_copy .+= step
        intercept += step
        k0 += 1
    end

    normalized_gradient = abs(sum(gradient(f, η_copy, y))) / n
    normalized_gradient <= 5 * strategy.gradient_tol && return intercept, k0
    error(
        "ExactStrategy failed to converge in $(strategy.maxit) iterations " *
            "(normalized intercept gradient = $normalized_gradient)",
    )
end

# Generic fallback: any one-step strategy counts as a single inner step.
function update_intercept_with_count(s::InterceptStrategy, f::LossFunction, intercept, η, y)
    return update_intercept(s, f, intercept, η, y), 1
end

function update_intercept_with_count(::NoIntercept, f::LossFunction, intercept, η, y)
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
    for _ in 0:armijo_max_backtracks
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
        ::UnguardedNewtonStrategy,
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

    return intercept .+ δ
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
    for _ in 0:s.max_backtracks
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
        strategy::ExactStrategy,
        f::LossFunction,
        intercept::AbstractVector{<:Real},
        η::AbstractMatrix{<:Real},
        y::AbstractVector{<:Integer},
    )
    new_intercept, _ = update_intercept_with_count(strategy, f, intercept, η, y)
    return new_intercept
end

function update_intercept_with_count(
        strategy::ExactStrategy,
        f::LossFunction,
        intercept::AbstractVector{<:Real},
        η::AbstractMatrix{<:Real},
        y::AbstractVector{<:Integer},
    )
    n = size(η, 1)
    η_copy = copy(η)
    intercept = copy(intercept)

    k0 = 0
    for _ in 1:strategy.maxit
        g = _intercept_grad(f, η_copy, y)

        if maximum(abs, g) / n <= strategy.gradient_tol
            return intercept, k0
        end

        H = _intercept_hess(f, η_copy, y)

        local direction
        try
            direction = -(H \ g)
        catch
            direction = -g ./ (f.lipschitz * n)
        end
        if any(!isfinite, direction) || !(dot(g, direction) < 0)
            direction = isfinite(f.lipschitz) && f.lipschitz > 0 ?
                -g ./ (f.lipschitz * n) :
                -g ./ n
        end

        f0 = loss(f, η_copy, y)
        slope = strategy.armijo_c * dot(g, direction)
        fp_tol = 4 * eps(Float64) * (1.0 + abs(f0))
        α = 1.0
        accepted = false
        for _ in 0:strategy.max_backtracks
            f_trial = loss(f, η_copy .+ α .* direction', y)
            if f_trial <= f0 + α * slope + fp_tol
                accepted = true
                break
            end
            α *= strategy.backtrack
        end
        accepted || error("ExactStrategy line search failed to find a descent step")

        step = α .* direction
        η_copy .+= step'
        intercept .+= step
        k0 += 1
    end

    normalized_gradient = maximum(abs, _intercept_grad(f, η_copy, y)) / n
    normalized_gradient <= strategy.gradient_tol && return intercept, k0
    error(
        "ExactStrategy failed to converge in $(strategy.maxit) iterations " *
            "(normalized intercept gradient = $normalized_gradient)",
    )
end
