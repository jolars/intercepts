using Intercepts
using Random
using ProjectRoot
using JLD2
using CSV
using DataFrames
using LinearAlgebra
using Statistics

import Intercepts: update_intercept

# Follow-up to sim-rho-centering.jl. This experiment tests whether the empirical
# T_G / T_N plateau is a local convergence-rate phenomenon. The strategy runs
# supply threshold crossings at 1e-4, 1e-6, and 1e-8, while Newton continues to
# 1e-12 so the shared dual reference is substantially tighter than the lowest
# reporting threshold.

const N = 500
const P = 1000
const S = 10
const REG = 0.05
const TOLERANCES = [1.0e-4, 1.0e-6, 1.0e-8]
const MAXIT = 120_000

# The first slice resolves the onset of the plateau in the existing design. The
# second changes the feature correlation and sweeps imbalance at a value of α
# already on the plateau. Duplicate cells are removed below.
const PRIMARY_CELLS = [
    (; ρ = 0.6, μ0, α) for μ0 in (0.5, 0.9) for α in (0.0, 0.1, 0.2, 0.3)
]
const SECONDARY_CELLS = [
    (; ρ = 0.2, μ0, α = 0.3) for μ0 in (0.5, 0.7, 0.9, 0.95)
]
const CELLS = unique(vcat(PRIMARY_CELLS, SECONDARY_CELLS))

"""
    TracedNewtonStrategy()

The scalar guarded Newton update used by `NewtonStrategy`, augmented with the
accepted Armijo scale from each pass. Keeping this instrumentation in the
experiment avoids changing the solver's public result type for one diagnostic.
"""
struct TracedNewtonStrategy <: InterceptStrategy
    accepted_scales::Vector{Float64}
end

TracedNewtonStrategy() = TracedNewtonStrategy(Float64[])

function update_intercept(
    strategy::TracedNewtonStrategy,
    f::LossFunction,
    intercept::Real,
    η::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
)
    grad = sum(gradient(f, η, y))
    hess = sum(hessian(f, η, y))

    if hess <= 0 || !isfinite(hess) || grad == 0
        push!(strategy.accepted_scales, 0.0)
        return intercept
    end

    direction = -grad / hess
    f0 = loss(f, η, y)
    armijo_c = 1.0e-4
    armijo_slope = armijo_c * grad * direction
    fp_tol = 4 * eps(Float64) * (1.0 + abs(f0))
    η_trial = similar(η)
    scale = 1.0

    for _ in 0:20
        @. η_trial = η + scale * direction
        f_trial = loss(f, η_trial, y)
        if f_trial <= f0 + scale * armijo_slope + fp_tol
            push!(strategy.accepted_scales, scale)
            return intercept + scale * direction
        end
        scale *= 0.5
    end

    push!(strategy.accepted_scales, 0.0)
    return intercept
end

function design(X_raw, α)
    centers = mean(X_raw; dims = 1)
    scales = stdm(X_raw, centers; corrected = false, dims = 1)
    scales[scales .== 0] .= 1.0
    return (X_raw .- (1 - α) .* centers) ./ scales
end

function rate_quantities(X, η, lossfun)
    w = weight(lossfun, η)
    H00 = sum(w)
    Hjj = vec(sum((X .^ 2) .* w; dims = 1))
    H0j = vec(sum(X .* w; dims = 1))
    ρ2 = (H0j .^ 2) ./ (H00 .* Hjj)
    return (; H00, barρ2 = sum(Hjj .* ρ2) / sum(Hjj))
end

passes_to(relgaps, tol) = let i = findfirst(g -> g <= tol, relgaps)
    isnothing(i) ? nothing : i - 1
end

function loggap_slope(relgaps; upper = 1.0e-4, lower = 1.0e-8)
    inds = findall(g -> isfinite(g) && lower <= g <= upper, relgaps)
    length(inds) >= 3 || return missing
    x = Float64.(inds)
    y = log.(relgaps[inds])
    x_centered = x .- mean(x)
    return dot(x_centered, y .- mean(y)) / dot(x_centered, x_centered)
end

function late_full_step_fraction(scales, relgaps; upper = 1.0e-4, lower = 1.0e-8)
    # relgaps[t] is measured before pass t, whose intercept scale is scales[t].
    n = min(length(scales), length(relgaps))
    inds = findall(t -> lower <= relgaps[t] <= upper, 1:n)
    isempty(inds) && return missing
    return mean(scales[inds] .== 1.0)
end

const CELLDIR = @projectroot("results", "rho-centering-diagnostics", "cells")
mkpath(CELLDIR)

cellpath(ρ, μ0, α) = joinpath(
    CELLDIR,
    "rho=$(ρ)_mu0=$(μ0)_alpha=$(α)_shared-v1.jld2",
)

records = Dict{String, Any}[]

for cell in CELLS
    (; ρ, μ0, α) = cell
    cached = cellpath(ρ, μ0, α)
    if isfile(cached)
        push!(records, JLD2.load(cached)["record"])
        @info "ρ=$ρ μ0=$μ0 α=$α (cached)"
        continue
    end

    Random.seed!(1)
    X_raw, y = generatedata(
        N,
        P;
        response = :binomial,
        μ0 = μ0,
        x_type = :normal,
        x_density = 0.9,
        ρ = ρ,
        s = S,
        amplitude = 1.0,
        means = :random,
    )
    X = design(X_raw, α)
    lossfun = LogisticLoss()
    L0 = lossfun.lipschitz * N

    traced_newton = TracedNewtonStrategy()
    Random.seed!(1)
    newton_result = cdsolver(
        X,
        y,
        REG;
        lossfun = lossfun,
        intercept_strategy = traced_newton,
        maxit = MAXIT,
        randomize = true,
        tol = 1.0e-12,
        normalization = :none,
    )

    reference_dual = maximum(newton_result.duals)
    reference_primal = minimum(newton_result.primals)
    reference_scale = max(abs(reference_primal), 1.0e-15)
    primal_stop = reference_dual + minimum(TOLERANCES) * reference_scale

    Random.seed!(1)
    gradient_result = cdsolver(
        X,
        y,
        REG;
        lossfun = lossfun,
        intercept_strategy = GradientStrategy(),
        maxit = MAXIT,
        randomize = true,
        tol = minimum(TOLERANCES),
        normalization = :none,
        primal_stop = primal_stop,
    )

    η_hat = X * newton_result.coef .+ newton_result.intercept
    q = rate_quantities(X, η_hat, lossfun)
    trajectories = [
        Dict{String, Any}(
            "primals" => result.primals,
            "duals" => result.duals,
            "gaps" => result.gaps,
            "relgaps" => result.relgaps,
        ) for result in (gradient_result, newton_result)
    ]
    suboptimality_against_shared_dual!(trajectories; instance_of = _ -> (ρ, μ0, α))
    gradient_relgaps = trajectories[1]["relgaps"]
    newton_relgaps = trajectories[2]["relgaps"]
    gradient_slope = loggap_slope(gradient_relgaps)
    newton_slope = loggap_slope(newton_relgaps)

    crossings = Dict{String, Any}()
    for tol in TOLERANCES
        suffix = string(Int(round(-log10(tol))))
        T_gradient = passes_to(gradient_relgaps, tol)
        T_newton = passes_to(newton_relgaps, tol)
        crossings["T_gradient_1e-$suffix"] = T_gradient
        crossings["T_newton_1e-$suffix"] = T_newton
        crossings["ratio_1e-$suffix"] =
            isnothing(T_gradient) || isnothing(T_newton) ? missing : T_gradient / T_newton
    end

    record = Dict{String, Any}(
        "rho" => ρ,
        "mu0" => μ0,
        "alpha" => α,
        "bar_rho2" => q.barρ2,
        "H00_over_L0" => q.H00 / L0,
        "asymptotic_ratio" => L0 / q.H00,
        "gradient_loggap_slope" => gradient_slope,
        "newton_loggap_slope" => newton_slope,
        "slope_ratio" => ismissing(gradient_slope) || ismissing(newton_slope) ?
            missing : newton_slope / gradient_slope,
        "newton_late_full_step_fraction" => late_full_step_fraction(
            traced_newton.accepted_scales,
            newton_relgaps,
        ),
        "gradient_final_relgap" => gradient_relgaps[end],
        "newton_final_relgap" => newton_relgaps[end],
        "gradient_relgaps" => gradient_relgaps,
        "newton_relgaps" => newton_relgaps,
        "newton_accepted_scales" => traced_newton.accepted_scales,
    )
    merge!(record, crossings)
    jldsave(cached; record)
    push!(records, record)

    slope_ratio_label = ismissing(record["slope_ratio"]) ?
        "missing" : string(round(record["slope_ratio"]; digits = 3))
    ratio_labels = getindex.(
        Ref(crossings),
        ["ratio_1e-4", "ratio_1e-6", "ratio_1e-8"],
    )
    full_step_fraction = record["newton_late_full_step_fraction"]
    @info "ρ=$ρ μ0=$μ0 α=$α barρ²=$(round(q.barρ2; digits = 4)) " *
        "L0/H00=$(round(L0 / q.H00; digits = 3)) " *
        "ratios=$ratio_labels " *
        "slope_ratio=$slope_ratio_label " *
        "full_steps=$full_step_fraction"
end

outfile = @projectroot("results", "rho-centering-diagnostics.jld2")
@save outfile records
println("Saved results to $outfile")

summary_columns = [
    "rho",
    "mu0",
    "alpha",
    "bar_rho2",
    "H00_over_L0",
    "asymptotic_ratio",
    "T_gradient_1e-4",
    "T_newton_1e-4",
    "ratio_1e-4",
    "T_gradient_1e-6",
    "T_newton_1e-6",
    "ratio_1e-6",
    "T_gradient_1e-8",
    "T_newton_1e-8",
    "ratio_1e-8",
    "gradient_loggap_slope",
    "newton_loggap_slope",
    "slope_ratio",
    "newton_late_full_step_fraction",
]
summary = DataFrame([(; (Symbol(k) => record[k] for k in summary_columns)...) for record in records])
sort!(summary, [:rho, :mu0, :alpha])
summary_file = @projectroot("results", "rho-centering-diagnostics.csv")
CSV.write(summary_file, summary)
println("Saved summary to $summary_file")
