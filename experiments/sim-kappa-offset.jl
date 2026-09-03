using Intercepts
using Random
using ProjectRoot
using CSV
using DataFrames
using LinearAlgebra
using SparseArrays
using Statistics

# Checks the closed form for the collective coupling under a common column
# offset (eq-kappa-offset) against the coupling measured at the converged
# solution, on the designs of sim-location-normalization.jl.
#
# Writing each active column as x_j = z_j + m*1 with z_j centered and m the
# offset in units of the column scale,
#
#   H_A0 = g + m*H_00*1,   H_AA = G + m(g1' + 1g') + m^2*H_00*11',
#
# with G the active block of the centered design and g = Z'W1 the weighted means
# of the centered columns. Dropping g leaves a rank-one update, and
# Sherman-Morrison gives kappa = theta/(1 + theta) with
# theta = m^2 * H_00 * 1'G^{-1}1. For near-orthogonal active columns
# 1'G^{-1}1 is about a/H_00, giving the coarser reading theta = m^2*a.

const N = 500
const P = 1000
const S = 10
const REG = 0.05
const AMPLITUDE = 1.0
const MAXIT = 40_000
const OPT_TOL = 1.0e-12
const ACTIVE_TOL = 1.0e-8

"""
    scaled_design(X_raw, centered)

Column-scaled design, with the column means removed when `centered`. Scales come
from `X_raw` either way, so the two arms differ only in location.
"""
function scaled_design(X_raw, centered::Bool)
    centers = mean(X_raw; dims = 1)
    scales = stdm(X_raw, centers; corrected = false, dims = 1)
    scales[scales .== 0] .= 1.0

    centered && return (Matrix(X_raw) .- centers) ./ scales

    X = copy(X_raw)
    for j in 1:size(X, 2)
        X[:, j] ./= scales[j]
    end
    return X
end

"""
    couplings(X_uncentered, X_centered, y, m)

Measured `κ` at the converged solution of the uncentered design, alongside the
closed form and its near-orthogonal reading. The common `1/n` cancels in every
ratio, so the unnormalized blocks are used throughout.
"""
function couplings(X_uncentered, X_centered, y, m)
    lossfun = LogisticLoss()

    Random.seed!(1)
    fit = cdsolver(
        X_uncentered, y, REG;
        lossfun = lossfun, intercept_strategy = NewtonStrategy(), maxit = MAXIT,
        randomize = true, normalization = :none, tol = OPT_TOL,
    )

    η = X_uncentered * fit.coef .+ fit.intercept
    w = weight(lossfun, η)
    active = findall(abs.(fit.coef) .> ACTIVE_TOL)
    Xa = Matrix(X_uncentered[:, active])
    Za = Matrix(X_centered[:, active])

    H00 = sum(w)
    c = vec(sum(Xa .* w; dims = 1))
    κ = dot(c, Symmetric(Xa' * (Xa .* w)) \ c) / H00

    G = Symmetric(Za' * (Za .* w))
    θ = m^2 * H00 * sum(G \ ones(length(active)))
    θ_orthogonal = m^2 * length(active)

    return (;
        a = length(active),
        kappa = κ,
        kappa_closed_form = θ / (1 + θ),
        kappa_orthogonal = θ_orthogonal / (1 + θ_orthogonal),
    )
end

records = DataFrame()

for q in (0.02, 0.05, 0.1, 0.3), μ0 in (0.5, 0.9)
    Random.seed!(1)
    X_raw = Float64.(sprand(Bool, N, P, q))
    β = zeros(P)
    β[Int.(round.(range(1, P; length = S)))] .= AMPLITUDE
    η = scaled_design(X_raw, true) * β .+ log(μ0 / (1 - μ0))
    y = Float64.(rand(N) .< (1 ./ (1 .+ exp.(-η))))

    m = sqrt(q / (1 - q)) # mean of an uncentered, scaled binary column
    r = couplings(scaled_design(X_raw, false), scaled_design(X_raw, true), y, m)
    push!(records, (; family = "binary", q = q, shift = NaN, μ0 = μ0, m = m, r...))
    @info "binary q=$q μ0=$μ0  κ=$(round(r.kappa; digits = 4))  closed form=$(round(r.kappa_closed_form; digits = 4))"
end

Random.seed!(2)
X_gauss, y_gauss = generatedata(
    N, P;
    response = :binomial, μ0 = 0.9, x_type = :normal, ρ = 0.6,
    s = S, amplitude = AMPLITUDE, means = nothing,
)
scales = stdm(X_gauss, mean(X_gauss; dims = 1); corrected = false, dims = 1)
scales[scales .== 0] .= 1.0
X_centered = (X_gauss .- mean(X_gauss; dims = 1)) ./ scales

for c in (0.25, 0.5, 1.0, 2.0)
    r = couplings((X_gauss .+ c) ./ scales, X_centered, y_gauss, c)
    push!(records, (; family = "gaussian", q = NaN, shift = c, μ0 = 0.9, m = c, r...))
    @info "gaussian shift=$c  κ=$(round(r.kappa; digits = 4))  closed form=$(round(r.kappa_closed_form; digits = 4))"
end

records.closed_form_error = abs.(records.kappa .- records.kappa_closed_form)
records.orthogonal_error = abs.(records.kappa .- records.kappa_orthogonal)

outfile = @projectroot("results", "kappa-offset.csv")
CSV.write(outfile, records)
println("Saved results to $outfile")
show(stdout, records; allrows = true, allcols = true)
println()
println("worst closed-form error: $(maximum(records.closed_form_error))")

# The paper quotes 0.003; fail loudly rather than let the claim drift.
@assert maximum(records.closed_form_error) <= 0.003
