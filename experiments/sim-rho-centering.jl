using Intercepts
using Random
using ProjectRoot
using JLD2
using LinearAlgebra
using Statistics

# Isolates the ρ_{0j}^2 axis of the Schur framing. At fixed μ_0 we sweep a
# centering fraction α ∈ [0, 1] that interpolates between the standardized
# design (α = 0, weighted ρ̄² ≈ 0) and the uncentered one (α = 1, ρ̄² ≈ 0.5,
# from generatedata's O(1) per-column means with `means = :random`):
#
#   X(α) = (X_raw - (1 - α) * colmeans) / colsds
#
# Column scales are fixed across α, so only the column offsets move. Shifting a
# column by a constant is absorbed exactly by the unpenalized intercept, so the
# lasso solution β* --- and hence η̂, H_00, and λ_max --- is invariant along the
# sweep. That makes H_00/L_0 exactly constant and leaves ρ̄² as the only moving
# quantity, which is what separates this experiment from the μ_0 sweep in
# sim-rate-gap.jl. We record H_00/L_0 per cell so the invariance is checkable
# rather than merely asserted.
#
# The eq-rate-gap leading-order prediction is (1 - (H_00/L_0) ρ̄²) / (1 - ρ̄²);
# we compare it to empirical pass counts T_G / T_N for the gradient and Newton
# strategies at a fixed tolerance.

const N = 500
const P = 1000
const S = 10
const REG = 0.05
const Μ0_GRID = [0.5, 0.9]
const ALPHA_GRID = collect(range(0.0, 1.0; length = 11))
const TOL_TARGET = 1.0e-6
const MAXIT = 40_000

"""
    design(X_raw, α)

Partially centered, fully scaled design: `α = 0` standardizes, `α = 1` keeps the
raw column means. Centers and scales come from `X_raw` and so are fixed across
the sweep.
"""
function design(X_raw, α)
    centers = mean(X_raw; dims = 1)
    scales = stdm(X_raw, centers; corrected = false, dims = 1)
    scales[scales .== 0] .= 1.0
    return (X_raw .- (1 - α) .* centers) ./ scales
end

function rate_quantities(X_used, η, lossfun)
    w = weight(lossfun, η)
    H00 = sum(w)
    Hjj = vec(sum((X_used .^ 2) .* w; dims = 1))
    H0j = vec(sum(X_used .* w; dims = 1))
    ρ2 = (H0j .^ 2) ./ (H00 .* Hjj)
    barρ2 = sum(Hjj .* ρ2) / sum(Hjj)
    return (; H00, Hjj, H0j, ρ2, barρ2, w)
end

passes_to(relgaps, tol) = let i = findfirst(g -> g ≤ tol, relgaps)
    i === nothing ? length(relgaps) : i
end

# The large-α cells run tens of thousands of passes, so the sweep is checkpointed
# per cell: each finished cell is written to results/rho-centering/cells/ and
# reloaded on a re-run instead of recomputed. The cells are scratch (gitignored,
# like the skglm-controlled ones); the assembled .jld2 below is the tracked
# artifact. Delete the cells directory to force a clean recomputation.
const CELLDIR = @projectroot("results", "rho-centering", "cells")
mkpath(CELLDIR)

cellpath(μ0, α) = joinpath(CELLDIR, "mu0=$(μ0)_alpha=$(round(α; digits = 2)).jld2")

records = Dict{String, Any}[]

for μ0 in Μ0_GRID
    Random.seed!(1)
    X_raw, y = generatedata(
        N,
        P;
        response = :binomial,
        μ0 = μ0,
        x_type = :normal,
        x_density = 0.9,
        ρ = 0.6,
        s = S,
        amplitude = 1.0,
        means = :random,
    )

    lossfun = LogisticLoss()
    L0 = lossfun.lipschitz * N # global Lipschitz on intercept: n * sup f''

    for α in ALPHA_GRID
        cached = cellpath(μ0, α)
        if isfile(cached)
            push!(records, JLD2.load(cached)["record"])
            @info "μ0=$(μ0) α=$(round(α; digits = 2))  (cached)"
            continue
        end

        X = design(X_raw, α)

        # Reference Newton run for η at convergence (used to evaluate H_jj, ρ²
        # at the iterate-level Hessian, matching how sim-rate-gap.jl reports
        # them). The solver sees exactly the matrix we measure ρ̄² on, so
        # normalization is off here.
        Random.seed!(1)
        newton_ref = cdsolver(
            X,
            y,
            REG;
            lossfun = lossfun,
            intercept_strategy = NewtonStrategy(),
            maxit = MAXIT,
            randomize = true,
            tol = 1.0e-12,
            normalization = :none,
            save_history = false,
        )

        η_hat = X * newton_ref.coef .+ newton_ref.intercept
        q = rate_quantities(X, η_hat, lossfun)

        # Cold-start intercept share |β_0^*| / ||β^*||, the quantity that drives
        # the R_0^2-dominance route into eq-rate-gap-asymptotic. Recorded so the
        # paper can check whether the asymptotic limit is reached *through* that
        # route along this sweep, or for some other reason.
        intercept_share = abs(newton_ref.intercept) / norm(newton_ref.coef)

        predicted_ratio = (1 - (q.H00 / L0) * q.barρ2) / (1 - q.barρ2)

        Random.seed!(1)
        grad_res = cdsolver(
            X,
            y,
            REG;
            lossfun = lossfun,
            intercept_strategy = GradientStrategy(),
            maxit = MAXIT,
            randomize = true,
            tol = TOL_TARGET,
            normalization = :none,
        )

        Random.seed!(1)
        newton_short = cdsolver(
            X,
            y,
            REG;
            lossfun = lossfun,
            intercept_strategy = NewtonStrategy(),
            maxit = MAXIT,
            randomize = true,
            tol = TOL_TARGET,
            normalization = :none,
        )

        T_grad = passes_to(grad_res.relgaps, TOL_TARGET)
        T_newt = passes_to(newton_short.relgaps, TOL_TARGET)
        grad_reached = any(g -> g ≤ TOL_TARGET, grad_res.relgaps)
        newt_reached = any(g -> g ≤ TOL_TARGET, newton_short.relgaps)

        empirical_ratio = T_grad / T_newt

        record = Dict{String, Any}(
            "μ0" => μ0,
            "α" => α,
            "n" => N,
            "p" => P,
            "s" => S,
            "reg" => REG,
            "tol" => TOL_TARGET,
            "L0" => L0,
            "H00" => q.H00,
            "H00_over_L0" => q.H00 / L0,
            "barρ2" => q.barρ2,
            "λmax" => grad_res.λmax,
            "intercept_star" => newton_ref.intercept,
            "coefnorm_star" => norm(newton_ref.coef),
            "intercept_share" => intercept_share,
            "asymptotic_ratio" => L0 / q.H00,
            "predicted_ratio" => predicted_ratio,
            "T_newton" => T_newt,
            "T_gradient" => T_grad,
            "empirical_ratio" => empirical_ratio,
            "newton_reached_tol" => newt_reached,
            "gradient_reached_tol" => grad_reached,
        )

        jldsave(cellpath(μ0, α); record)
        push!(records, record)

        @info "μ0=$(μ0) α=$(round(α; digits = 2))  H00/L0=$(round(q.H00 / L0; sigdigits = 4))  ρ̄²=$(round(q.barρ2; sigdigits = 3))  predicted=$(round(predicted_ratio; sigdigits = 3))  empirical=$(round(empirical_ratio; sigdigits = 3))  L0/H00=$(round(L0 / q.H00; sigdigits = 3))  |β0|/‖β‖=$(round(intercept_share; sigdigits = 3))  (T_N=$(T_newt), T_G=$(T_grad))"
    end
end

outfile = @projectroot("results", "rho-centering.jld2")
@save outfile records
println("Saved results to $outfile")
