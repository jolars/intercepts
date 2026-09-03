using Intercepts
using Random
using ProjectRoot
using JLD2
using CSV
using DataFrames
using LinearAlgebra
using SparseArrays
using Statistics

# Does feature *location* interact with the intercept update strategy, and does
# the normalization choice therefore matter?
#
# sim-rho-centering.jl already answers a version of this on a dense Gaussian
# design with `means = :random`: relaxing the centering raises the weighted ρ̄²
# and pushes T_G/T_N from about one to the L_0/H_00 plateau. Those column
# offsets are *incoherent* --- randn signs --- and the sweep tops out near
# ρ̄² ≈ 0.5. Two things are left open, and both are the practitioner's case:
#
#   1. Coherence. ρ̄² averages squared per-coordinate correlations and is blind
#      to the sign pattern of the offsets; κ = H_0A H_AA^{-1} H_A0 / H_00 sums
#      through H_AA^{-1} and is not. Binary 0/1 columns have all-positive means,
#      every one of them aligned with the intercept's 1_n column, so they should
#      drive κ toward one harder than signed offsets at the same ρ̄².
#   2. Density. An uncentered-but-scaled binary column of density q has mean
#      sqrt(q/(1-q)), which diverges as q → 0. Rare binary features --- exactly
#      the sparse, large-n setting where people skip centering to keep the design
#      sparse --- sit in the corner the α sweep cannot reach.
#
# Family B (headline) sweeps binary density × centering × μ_0. Family A isolates
# the coherence mechanism on a dense design by shifting every column by the same
# constant (coherent) or by ±constant (incoherent).
#
# The comparison is controlled: shifting a column by a constant is absorbed
# exactly by the unpenalized intercept, so β*, η̂, H_00 and λ_max are invariant
# across the centering arms, and κ / ρ̄² are the only quantities that move. Each
# cell records the realized discrepancy in β* and λ_max so the invariance is
# checked rather than asserted; a hard check over all cells runs at the end.
#
# Normalization note: Intercepts' :standardize centers sparse designs too --- it
# skips the subtraction in normalizefeatures but reinstates it exactly through
# cdsolver's x_sparse_offset. The "scale but do not center" mode practitioners
# use is therefore not a package option, so we build both designs here and pass
# normalization = :none, as sim-rho-centering.jl does.
#
# We report passes, not wall-clock: the centered arm is dense and the uncentered
# arm is sparse, so times are not comparable, and Section 4.5 already uses passes
# to decouple the curvature prediction from BLAS effects.

const N = 500
const P = 1000
const S = 10
const AMPLITUDE = 1.0
const REG = 0.05
const TOL_TARGET = 1.0e-6
const OPT_TOL = 1.0e-12
const MAXIT = 40_000
const ACTIVE_TOL = 1.0e-8

# Family B: sparse binary designs. ρ = 0 on purpose --- generatedata's binary
# correlation copies the previous column, which perturbs the realized density and
# would confound the q axis.
const DENSITY_GRID = [0.02, 0.05, 0.1, 0.3]
const Μ0_GRID = [0.5, 0.9]

# Family A: dense Gaussian, same shift magnitude applied coherently or not.
const SHIFT_GRID = [0.0, 0.25, 0.5, 1.0, 2.0]
const SHIFT_Μ0 = 0.9

const CELLDIR = @projectroot("results", "location-normalization", "cells")
mkpath(CELLDIR)

"""
    scaled_design(X_raw, centered)

Column-scaled design. `centered = true` subtracts the column means and densifies;
`centered = false` keeps the raw locations and, for a sparse `X_raw`, keeps the
sparsity. Scales come from `X_raw` either way, so the two arms differ only in
location.
"""
function scaled_design(X_raw, centered::Bool)
    centers = mean(X_raw; dims = 1)
    scales = stdm(X_raw, centers; corrected = false, dims = 1)
    scales[scales .== 0] .= 1.0

    if centered
        return (Matrix(X_raw) .- centers) ./ scales
    end

    X = copy(X_raw)
    for j in 1:size(X, 2)
        # Column-wise, not a broadcast against a row vector: broadcasting would
        # densify the sparse arm and defeat the point of the experiment.
        X[:, j] ./= scales[j]
    end
    return X
end

"""
    binary_problem(q, μ0)

Sparse binary design of density `q` with a logistic response whose marginal mean
is `μ0`.

`generatedata`'s binary branch cannot be used here. It forms the linear predictor
from the *raw* 0/1 columns, and those are non-negative, so the signal adds an
offset of `s * q * amplitude` to η: the realized response mean climbs with `q`
(0.56 → 0.96 at μ0 = 0.5, and all-ones at q = 0.3, μ0 = 0.9, where λmax collapses
to zero). That would confound the density axis with the imbalance axis, when the
whole point is to move location while holding H_00/L_0 fixed.

Building η from the *standardized* design instead makes the signal mean-zero, so
`μ0` is the true marginal mean --- the convention the paper uses for its
standardized Gaussian designs. Both centering arms share the returned `y`.
"""
function binary_problem(q::Real, μ0::Real)
    X_raw = Float64.(sprand(Bool, N, P, Float64(q)))

    β = zeros(P)
    β[Int.(round.(range(1, P; length = S)))] .= AMPLITUDE

    η = scaled_design(X_raw, true) * β .+ log(μ0 / (1 - μ0))
    y = Float64.(rand(N) .< (1 ./ (1 .+ exp.(-η))))

    # A degenerate response makes λmax zero and the unpenalized fit divergent;
    # fail here rather than thousands of passes later.
    positives = sum(y)
    0 < positives < N ||
        error("degenerate response at q=$q, μ0=$μ0: $(Int(positives)) positives of $N")

    return X_raw, y
end

"""
    rate_quantities(X_used, η, lossfun)

Iterate-level intercept curvature `H00`, per-coordinate `H_jj`, cross-curvature
`H_0j`, and the curvature-weighted mean squared coupling `ρ̄²`.
"""
function rate_quantities(X_used, η, lossfun)
    w = weight(lossfun, η)
    H00 = sum(w)
    Hjj = vec(sum((X_used .^ 2) .* w; dims = 1))
    H0j = vec(sum(X_used .* w; dims = 1))
    ρ2 = (H0j .^ 2) ./ (H00 .* Hjj)
    barρ2 = sum(Hjj .* ρ2) / sum(Hjj)
    return (; H00, Hjj, H0j, ρ2, barρ2, w)
end

"""
    collective_coupling(X, η, coef, lossfun)

The active-set collective coupling κ = H_0A H_AA^{-1} H_A0 / H_00. The common
1/n factor cancels, so the unnormalized blocks are used directly. Returns
`(κ, n_active)`.
"""
function collective_coupling(X, η, coef, lossfun)
    active = findall(abs.(coef) .> ACTIVE_TOL)
    isempty(active) && return (NaN, 0)

    X_active = Matrix(X[:, active])
    w = weight(lossfun, η)
    h = sum(w)
    c = vec(sum(X_active .* w; dims = 1))
    A = Symmetric(X_active' * (X_active .* w))

    κ = dot(c, A \ c) / h
    return (κ, length(active))
end

passes_to(relgaps, tol) = let i = findfirst(g -> g ≤ tol, relgaps)
    i === nothing ? length(relgaps) : i - 1
end

solve(X, y, lossfun, strategy; kwargs...) = begin
    Random.seed!(1)
    cdsolver(
        X, y, REG;
        lossfun = lossfun,
        intercept_strategy = strategy,
        maxit = MAXIT,
        randomize = true,
        normalization = :none,
        kwargs...,
    )
end

"""
    run_arm(X, y, lossfun, L0, meta)

Solve one design with the gradient and Newton strategies, score both against a
shared dual, and return the record. `meta` supplies the cell's identifying keys.
"""
function run_arm(X, y, lossfun, L0, meta)
    reference = solve(X, y, lossfun, NewtonStrategy(); tol = OPT_TOL)

    η_hat = X * reference.coef .+ reference.intercept
    rates = rate_quantities(X, η_hat, lossfun)
    κ, n_active = collective_coupling(X, η_hat, reference.coef, lossfun)

    reference_scale = max(abs(minimum(reference.primals)), 1.0e-15)
    primal_stop = maximum(reference.duals) + TOL_TARGET * reference_scale

    gradient_run = solve(
        X, y, lossfun, GradientStrategy();
        tol = TOL_TARGET, primal_stop = primal_stop,
    )
    newton_run = solve(
        X, y, lossfun, NewtonStrategy();
        tol = TOL_TARGET, primal_stop = primal_stop,
    )

    trajectories = [
        Dict{String, Any}(
            "primals" => result.primals,
            "duals" => result.duals,
            "gaps" => result.gaps,
            "relgaps" => result.relgaps,
        ) for result in (gradient_run, newton_run, reference)
    ]
    suboptimality_against_shared_dual!(trajectories; instance_of = _ -> 1)

    gradient_relgaps = trajectories[1]["relgaps"]
    newton_relgaps = trajectories[2]["relgaps"]
    T_gradient = passes_to(gradient_relgaps, TOL_TARGET)
    T_newton = passes_to(newton_relgaps, TOL_TARGET)

    record = Dict{String, Any}(
        "n" => N,
        "p" => P,
        "s" => S,
        "reg" => REG,
        "tol" => TOL_TARGET,
        "mean_y" => mean(y),
        "L0" => L0,
        "H00" => rates.H00,
        "H00_over_L0" => rates.H00 / L0,
        "barρ2" => rates.barρ2,
        "kappa" => κ,
        "n_active" => n_active,
        "λmax" => reference.λmax,
        "intercept_star" => reference.intercept,
        "coefnorm_star" => norm(reference.coef),
        "intercept_share" => abs(reference.intercept) / norm(reference.coef),
        "asymptotic_ratio" => L0 / rates.H00,
        "T_gradient" => T_gradient,
        "T_newton" => T_newton,
        "empirical_ratio" => T_gradient / T_newton,
        "gradient_reached_tol" => any(g -> g ≤ TOL_TARGET, gradient_relgaps),
        "newton_reached_tol" => any(g -> g ≤ TOL_TARGET, newton_relgaps),
    )
    merge!(record, meta)
    return record, reference
end

"""
    invariance!(records, reference_a, reference_b)

Record how far the two centering arms' solutions drift apart. A constant column
shift is absorbed by the unpenalized intercept, so `β*` and `λmax` must agree;
any real discrepancy means the two arms are not the same problem and the cell's
ratios are meaningless.
"""
function invariance!(records, reference_a, reference_b)
    β_gap = norm(reference_a.coef - reference_b.coef, Inf)
    λ_gap = abs(reference_a.λmax - reference_b.λmax) / max(reference_a.λmax, 1.0e-15)

    for record in records
        record["beta_discrepancy"] = β_gap
        record["lambdamax_discrepancy"] = λ_gap
    end

    if β_gap > 1.0e-5 || λ_gap > 1.0e-8
        @warn "Centering arms disagree" β_gap λ_gap cell = records[1]["cell"]
    end
    return nothing
end

cellpath(name) = joinpath(CELLDIR, "$(name)_v1.jld2")

"""
    cell!(records, name, build)

Run one `(design family, parameter, μ_0)` cell --- both centering arms --- or
reload it from the per-cell checkpoint. `build(centered)` returns the design.
"""
function cell!(records, name, μ0, y, lossfun, L0, meta, build)
    cached = cellpath(name)
    if isfile(cached)
        append!(records, JLD2.load(cached)["cell_records"])
        @info "$(name)  (cached)"
        return nothing
    end

    cell_records = Dict{String, Any}[]
    references = []

    for centered in (true, false)
        arm_meta = merge(
            Dict{String, Any}("cell" => name, "μ0" => μ0, "centered" => centered),
            meta,
        )
        record, reference = run_arm(build(centered), y, lossfun, L0, arm_meta)
        push!(cell_records, record)
        push!(references, reference)

        @info "$(name) centered=$(centered)  " *
            "κ=$(round(record["kappa"]; sigdigits = 3))  " *
            "ρ̄²=$(round(record["barρ2"]; sigdigits = 3))  " *
            "H00/L0=$(round(record["H00_over_L0"]; sigdigits = 3))  " *
            "L0/H00=$(round(record["asymptotic_ratio"]; sigdigits = 3))  " *
            "T_G/T_N=$(round(record["empirical_ratio"]; sigdigits = 3))  " *
            "(T_G=$(record["T_gradient"]), T_N=$(record["T_newton"]))"
    end

    invariance!(cell_records, references[1], references[2])

    jldsave(cached; cell_records)
    append!(records, cell_records)
    return nothing
end

records = Dict{String, Any}[]
lossfun = LogisticLoss()
L0 = lossfun.lipschitz * N # global Lipschitz on the intercept: n * sup f''

# --- Family B: sparse binary designs -----------------------------------------

for μ0 in Μ0_GRID, q in DENSITY_GRID
    Random.seed!(1)
    X_raw, y = binary_problem(q, μ0)

    meta = Dict{String, Any}(
        "family" => "binary",
        "density" => q,
        "shift" => NaN,
        "coherence" => "positive",
        # Location-to-scale ratio of an uncentered, scaled binary column.
        "mean_over_sd" => sqrt(q / (1 - q)),
    )

    cell!(
        records, "binary_q=$(q)_mu0=$(μ0)", μ0, y, lossfun, L0, meta,
        centered -> scaled_design(X_raw, centered),
    )
end

# --- Family A: dense Gaussian, coherent versus incoherent offsets -------------

Random.seed!(2)
X_gauss, y_gauss = generatedata(
    N, P;
    response = :binomial,
    μ0 = SHIFT_Μ0,
    x_type = :normal,
    ρ = 0.6,
    s = S,
    amplitude = 1.0,
    means = nothing,
)

gauss_scales = stdm(X_gauss, mean(X_gauss; dims = 1); corrected = false, dims = 1)
gauss_scales[gauss_scales .== 0] .= 1.0

Random.seed!(3)
const SIGNS = rand([-1.0, 1.0], P)

for coherence in ("positive", "signed"), shift in SHIFT_GRID
    m = coherence == "positive" ? ones(P) : SIGNS

    meta = Dict{String, Any}(
        "family" => "gaussian",
        "density" => NaN,
        "shift" => shift,
        "coherence" => coherence,
        "mean_over_sd" => shift,
    )

    cell!(
        records,
        "gaussian_$(coherence)_shift=$(shift)_mu0=$(SHIFT_Μ0)",
        SHIFT_Μ0, y_gauss, lossfun, L0, meta,
        # The centered arm removes the shift entirely, so it is the same design
        # for every `shift` --- a repeated baseline that also re-checks the
        # invariance at each level.
        centered -> centered ?
            (X_gauss .- mean(X_gauss; dims = 1)) ./ gauss_scales :
            (X_gauss .+ shift .* m') ./ gauss_scales,
    )
end

# --- Save ---------------------------------------------------------------------

outfile = @projectroot("results", "location-normalization.jld2")
@save outfile records
println("Saved results to $outfile")

summary_columns = [
    "family", "coherence", "density", "shift", "mean_over_sd", "μ0", "centered",
    "mean_y", "H00_over_L0", "barρ2", "kappa", "n_active", "intercept_share",
    "T_gradient", "T_newton", "empirical_ratio", "asymptotic_ratio",
    "gradient_reached_tol", "newton_reached_tol",
    "beta_discrepancy", "lambdamax_discrepancy",
]
summary = DataFrame([
    (; (Symbol(k) => record[k] for k in summary_columns)...) for record in records
])
sort!(summary, [:family, :coherence, :μ0, :density, :shift, :centered])

summary_file = @projectroot("results", "location-normalization.csv")
CSV.write(summary_file, summary)
println("Saved summary to $summary_file")

show(stdout, summary; allrows = true, allcols = true)
println()

# The invariance is what makes the centering arms the same problem. Fail loudly
# rather than quietly reporting ratios between two different optimizations.
worst_β = maximum(record["beta_discrepancy"] for record in records)
worst_λ = maximum(record["lambdamax_discrepancy"] for record in records)
println("worst β* discrepancy across centering arms: $worst_β")
println("worst λmax discrepancy across centering arms: $worst_λ")
@assert worst_β <= 1.0e-5 "centering arms disagree on β*: $worst_β"
@assert worst_λ <= 1.0e-8 "centering arms disagree on λmax: $worst_λ"
@assert all(
    record -> isnan(record["kappa"]) || 0 <= record["kappa"] < 1 + 1.0e-8, records
) "κ outside [0, 1)"
@assert all(record -> 0 <= record["barρ2"] <= 1 + 1.0e-8, records) "ρ̄² outside [0, 1]"
