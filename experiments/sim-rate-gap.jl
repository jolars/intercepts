using Intercepts
using Random
using ProjectRoot
using JLD2
using LinearAlgebra
using Statistics

# Quantitative rate gap experiment: sweep μ0 and compare empirical pass counts
# to reach a fixed relative-gap tolerance against the predicted ratio
# (1 - (H_00/L_0) ρ̄²) / (1 - ρ̄²), with weights and curvatures evaluated at
# the Newton-strategy converged iterate.

const N = 500
const P = 1000
const S = 10
const REG = 0.05
const Μ0_GRID = [0.5, 0.7, 0.9, 0.95, 0.99]
const TOL_TARGET = 1.0e-6
const MAXIT = 5000

function standardize_features(X)
    centers = mean(X; dims = 1)
    scales = stdm(X, centers; corrected = false, dims = 1)
    scales[scales.==0] .= 1.0
    return (X .- centers) ./ scales, centers, scales
end

function rate_quantities(X_std, η, lossfun)
    w = weight(lossfun, η)
    H00 = sum(w)
    Hjj = vec(sum((X_std .^ 2) .* w; dims = 1))
    H0j = vec(sum(X_std .* w; dims = 1))
    ρ2 = (H0j .^ 2) ./ (H00 .* Hjj)
    barρ2 = sum(Hjj .* ρ2) / sum(Hjj)
    return (; H00, Hjj, H0j, ρ2, barρ2, w)
end

passes_to(relgaps, tol) = let i = findfirst(g -> g ≤ tol, relgaps)
    i === nothing ? length(relgaps) : i
end

records = Dict{String,Any}[]

for μ0 in Μ0_GRID
    Random.seed!(1)
    X, y = generatedata(
        N, P;
        response = :binomial, μ0 = μ0, x_type = :normal,
        x_density = 0.9, ρ = 0.6, s = S, amplitude = 1.0, means = :random,
    )

    # Reference Newton run: same parameters as cdsolver uses internally, but we
    # use save_history=true to recover η at convergence on the standardized
    # design.
    X_std, x_centers, x_scales = standardize_features(X)
    lossfun = LogisticLoss()
    L0 = lossfun.lipschitz * N  # global Lipschitz on intercept: n * sup f''

    Random.seed!(1)
    newton_res = cdsolver(
        X, y, REG;
        lossfun = lossfun,
        intercept_strategy = NewtonStrategy(),
        maxit = MAXIT,
        randomize = true,
        tol = 1.0e-12,
        save_history = false,
    )

    # Reconstruct η at the Newton solution. cdsolver returns coef on the
    # original X scale; η = X β + β0 is scale-invariant.
    η_hat = X * newton_res.coef .+ newton_res.intercept
    q = rate_quantities(X_std, η_hat, lossfun)

    # β on the standardized design: invert the rescaling done by
    # `rescalecoefs` in src/normalize.jl. With cold start β = 0, the CD-bound
    # distances are R_0² = (β₀⋆)², R_j² = (β_j⋆)² in standardized coords.
    β_std = newton_res.coef .* vec(x_scales)
    β0_std = newton_res.intercept + dot(vec(x_centers), newton_res.coef)
    @assert isapprox(X_std * β_std .+ β0_std, η_hat; atol = 1.0e-8)
    R0_sq = β0_std^2
    Rj_sq = β_std .^ 2

    # Full @eq-rate-gap ratio evaluated at the Newton-converged iterate.
    num_intercept = L0 * R0_sq
    den_intercept = q.H00 * R0_sq
    num_slope = sum(q.Hjj .* (1 .- (q.H00 / L0) .* q.ρ2) .* Rj_sq)
    den_slope = sum(q.Hjj .* (1 .- q.ρ2) .* Rj_sq)
    full_ratio = (num_intercept + num_slope) / (den_intercept + den_slope)

    predicted_ratio = (1 - (q.H00 / L0) * q.barρ2) / (1 - q.barρ2)

    # Empirical pass counts at TOL_TARGET.
    Random.seed!(1)
    grad_res = cdsolver(
        X, y, REG;
        lossfun = lossfun,
        intercept_strategy = GradientStrategy(),
        maxit = MAXIT,
        randomize = true,
        tol = TOL_TARGET,
    )

    Random.seed!(1)
    newton_short = cdsolver(
        X, y, REG;
        lossfun = lossfun,
        intercept_strategy = NewtonStrategy(),
        maxit = MAXIT,
        randomize = true,
        tol = TOL_TARGET,
    )

    T_grad = passes_to(grad_res.relgaps, TOL_TARGET)
    T_newt = passes_to(newton_short.relgaps, TOL_TARGET)
    grad_reached = any(g -> g ≤ TOL_TARGET, grad_res.relgaps)
    newt_reached = any(g -> g ≤ TOL_TARGET, newton_short.relgaps)

    empirical_ratio = T_grad / T_newt

    push!(records, Dict{String,Any}(
        "μ0" => μ0,
        "n" => N,
        "p" => P,
        "s" => S,
        "reg" => REG,
        "tol" => TOL_TARGET,
        "L0" => L0,
        "H00" => q.H00,
        "H00_over_L0" => q.H00 / L0,
        "barρ2" => q.barρ2,
        "R0_sq" => R0_sq,
        "sum_Rj_sq" => sum(Rj_sq),
        "num_intercept" => num_intercept,
        "den_intercept" => den_intercept,
        "num_slope" => num_slope,
        "den_slope" => den_slope,
        "full_ratio" => full_ratio,
        "predicted_ratio" => predicted_ratio,
        "T_newton" => T_newt,
        "T_gradient" => T_grad,
        "empirical_ratio" => empirical_ratio,
        "newton_reached_tol" => newt_reached,
        "gradient_reached_tol" => grad_reached,
    ))

    @info "μ0=$(μ0)  H00/L0=$(round(q.H00/L0; sigdigits=3))  ρ̄²=$(round(q.barρ2; sigdigits=3))  full=$(round(full_ratio; sigdigits=3))  L0/H00=$(round(L0/q.H00; sigdigits=3))  empirical=$(round(empirical_ratio; sigdigits=3))  (T_N=$(T_newt), T_G=$(T_grad))"
end

outfile = @projectroot("results", "rate-gap.jld2")
@save outfile records
