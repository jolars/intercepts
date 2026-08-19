using Intercepts
using Random
using ProjectRoot
using JLD2
using CSV
using DataFrames
using LinearAlgebra
using Statistics

# Frozen-Hessian follow-up to sim-rho-centering-diagnostics.jl. For each
# coupling-dominated cell, this script computes the active-set multiple
# coupling κ, the spectral radius of a cyclic coordinate pass, the spectral
# radius of the mean permuted-pass matrix, and the top Lyapunov exponent of
# random products of permuted-pass matrices.

const N = 500
const P = 1000
const S = 10
const REG = 0.05
const MAXIT = 120_000
const OPT_TOL = 1.0e-10
const ACTIVE_TOL = 1.0e-8
const N_MEAN_PERMUTATIONS = 500
const N_LYAPUNOV_PASSES = 20_000
const CELLS = vcat(
    [(; ρ = 0.6, μ0, α) for μ0 in (0.5, 0.9) for α in (0.0, 0.1, 0.2, 0.3)],
    [(; ρ = 0.2, μ0, α = 0.3) for μ0 in (0.5, 0.7, 0.9, 0.95)],
)

function design(X_raw, α)
    centers = mean(X_raw; dims = 1)
    scales = stdm(X_raw, centers; corrected = false, dims = 1)
    scales[scales .== 0] .= 1.0
    return (X_raw .- (1 - α) .* centers) ./ scales
end

function active_hessian(X, η, coef, lossfun)
    active = findall(abs.(coef) .> ACTIVE_TOL)
    X_active = X[:, active]
    w = weight(lossfun, η)
    h = sum(w)
    c = vec(sum(X_active .* w; dims = 1))
    A = X_active' * (X_active .* w)
    H = [h c'; c A]
    return (; active, h, c, A, H)
end

function coefficient_pass_matrix(H, order)
    dimension = size(H, 1)
    pass = Matrix{Float64}(I, dimension, dimension)
    for active_j in order
        j = active_j + 1
        update = Matrix{Float64}(I, dimension, dimension)
        update[j, :] .-= H[j, :] ./ H[j, j]
        pass = update * pass
    end
    return pass
end

function intercept_update_matrix(H, denominator)
    update = Matrix{Float64}(I, size(H, 1), size(H, 2))
    update[1, :] .-= H[1, :] ./ denominator
    return update
end

spectral_radius(M) = maximum(abs, eigvals(M))

function mean_pass_matrices(H, h, L0, rng)
    m = size(H, 1) - 1
    mean_newton = zeros(size(H))
    mean_gradient = zeros(size(H))
    intercept_newton = intercept_update_matrix(H, h)
    intercept_gradient = intercept_update_matrix(H, L0)

    for _ in 1:N_MEAN_PERMUTATIONS
        coefficient_pass = coefficient_pass_matrix(H, randperm(rng, m))
        mean_newton .+= intercept_newton * coefficient_pass
        mean_gradient .+= intercept_gradient * coefficient_pass
    end

    mean_newton ./= N_MEAN_PERMUTATIONS
    mean_gradient ./= N_MEAN_PERMUTATIONS
    return (; mean_newton, mean_gradient)
end

function apply_coordinate_pass!(state, H, order)
    for active_j in order
        j = active_j + 1
        state[j] -= dot(H[j, :], state) / H[j, j]
    end
    return state
end

function lyapunov_log_rate(H, denominator, rng)
    m = size(H, 1) - 1
    state = randn(rng, m + 1)
    state ./= norm(state)
    accumulated_log_norm = 0.0
    burnin = N_LYAPUNOV_PASSES ÷ 10

    for pass in 1:N_LYAPUNOV_PASSES
        apply_coordinate_pass!(state, H, randperm(rng, m))
        state[1] -= dot(H[1, :], state) / denominator
        state_norm = norm(state)
        state ./= state_norm
        if pass > burnin
            accumulated_log_norm += log(state_norm)
        end
    end

    return accumulated_log_norm / (N_LYAPUNOV_PASSES - burnin)
end

const CELLDIR = @projectroot("results", "rho-frozen-hessian", "cells")
mkpath(CELLDIR)

cellpath(ρ, μ0, α) = joinpath(CELLDIR, "rho=$(ρ)_mu0=$(μ0)_alpha=$(α).jld2")

records = Dict{String, Any}[]

for (; ρ, μ0, α) in CELLS
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

    Random.seed!(1)
    optimum = cdsolver(
        X,
        y,
        REG;
        lossfun = lossfun,
        intercept_strategy = NewtonStrategy(),
        maxit = MAXIT,
        randomize = true,
        tol = OPT_TOL,
        normalization = :none,
    )
    η = X * optimum.coef .+ optimum.intercept
    active = active_hessian(X, η, optimum.coef, lossfun)
    (; h, c, A, H) = active
    q = h / L0
    κ = dot(c, A \ c) / h

    cyclic_coefficients = coefficient_pass_matrix(H, 1:length(active.active))
    cyclic_newton = intercept_update_matrix(H, h) * cyclic_coefficients
    cyclic_gradient = intercept_update_matrix(H, L0) * cyclic_coefficients

    mean_rng = MersenneTwister(2)
    mean_passes = mean_pass_matrices(H, h, L0, mean_rng)

    newton_log_rate = lyapunov_log_rate(H, h, MersenneTwister(3))
    gradient_log_rate = lyapunov_log_rate(H, L0, MersenneTwister(3))
    block_newton_rate = κ
    block_gradient_rate = 1 - q * (1 - κ)

    record = Dict{String, Any}(
        "rho" => ρ,
        "mu0" => μ0,
        "alpha" => α,
        "active_size" => length(active.active),
        "q" => q,
        "L0_over_H00" => inv(q),
        "kappa" => κ,
        "block_newton_rate" => block_newton_rate,
        "block_gradient_rate" => block_gradient_rate,
        "block_log_rate_ratio" => log(block_newton_rate) / log(block_gradient_rate),
        "cyclic_newton_radius" => spectral_radius(cyclic_newton),
        "cyclic_gradient_radius" => spectral_radius(cyclic_gradient),
        "cyclic_log_rate_ratio" =>
            log(spectral_radius(cyclic_newton)) / log(spectral_radius(cyclic_gradient)),
        "mean_newton_radius" => spectral_radius(mean_passes.mean_newton),
        "mean_gradient_radius" => spectral_radius(mean_passes.mean_gradient),
        "mean_log_rate_ratio" =>
            log(spectral_radius(mean_passes.mean_newton)) /
            log(spectral_radius(mean_passes.mean_gradient)),
        "random_newton_log_rate" => newton_log_rate,
        "random_gradient_log_rate" => gradient_log_rate,
        "random_log_rate_ratio" => newton_log_rate / gradient_log_rate,
        "final_relgap" => optimum.relgaps[end],
    )
    jldsave(cached; record)
    push!(records, record)

    active_size = record["active_size"]
    block_ratio = round(record["block_log_rate_ratio"]; digits = 3)
    cyclic_ratio = round(record["cyclic_log_rate_ratio"]; digits = 3)
    random_ratio = round(record["random_log_rate_ratio"]; digits = 3)
    @info "ρ=$ρ μ0=$μ0 α=$α active=$active_size " *
        "κ=$(round(κ; digits = 5)) L0/H00=$(round(inv(q); digits = 3)) " *
        "block=$block_ratio cyclic=$cyclic_ratio random=$random_ratio"
end

outfile = @projectroot("results", "rho-frozen-hessian.jld2")
@save outfile records

summary = DataFrame(records)
select!(summary, Not(:final_relgap))
sort!(summary, [:rho, :mu0, :alpha])
summary_file = @projectroot("results", "rho-frozen-hessian.csv")
CSV.write(summary_file, summary)
println("Saved results to $outfile and $summary_file")
