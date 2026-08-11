using Intercepts
using Random
using LinearAlgebra
using Statistics
using ProjectRoot
using CSV
using DataFrames
using JSON3

# Problem family for the skglm pre-fix vs post-fix controlled comparison.
# Mirrors the (μ0, reg, seed) grid of sim-mu-reg-heatmap.jl so the resulting
# pass-count-ratio heatmap (@fig-skglm-controlled) reads as a direct companion
# to @fig-mu-reg-gradient. Each cell is a standardized logistic problem; the
# Python driver experiments/sim-skglm-controlled.py loads them by id and runs
# skglm AndersonCD at two pinned commits of /home/jola/projects/skglm:
#   - b03644fe (pre-fix, gradient intercept update; matches skglm 0.5)
#   - 8f9cbd77 (post-fix, Newton intercept update; the merge of PR #337)
#
# Standardization and λ_max convention match sim-real-problem.jl exactly so
# both experiments speak the same penalty scale.
#
# The per-cell cells/cell_<id>/{X.csv,y.csv,meta.json} are scratch (large and
# reproducible from the seeds in index.csv); .gitignore handles them. Only
# index.csv and the eventual results.csv ship with the repo.

const N = 500
const P = 1000
const S = 10
const ρ_ = 0.6

const μ0_grid = [0.5, 0.7, 0.9, 0.95, 0.99]
const reg_grid = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
const seed_grid = [1, 2, 3]

outdir = @projectroot("results", "skglm-controlled")
cells_dir = joinpath(outdir, "cells")
mkpath(cells_dir)

index_rows = Vector{NamedTuple}()
cell_id = 0

for seed in seed_grid, μ0 in μ0_grid, reg in reg_grid
    global cell_id += 1
    Random.seed!(seed)

    X_raw, y01 = generatedata(
        N,
        P;
        response = :binomial,
        μ0 = μ0,
        x_type = :normal,
        ρ = ρ_,
        s = S,
        amplitude = 1.0,
    )

    x_centers = vec(mean(X_raw; dims = 1))
    x_scales = vec(std(X_raw; dims = 1, corrected = false))
    x_scales[x_scales .== 0] .= 1.0
    X = (X_raw .- x_centers') ./ x_scales'

    y_pm = 2.0 .* y01 .- 1.0
    ybar = mean(y01)

    lambda_max = norm(X' * (y01 .- ybar), Inf) / N
    lambda = reg * lambda_max

    cell_name = "cell_" * lpad(cell_id, 3, '0')
    cell_path = joinpath(cells_dir, cell_name)
    mkpath(cell_path)

    CSV.write(joinpath(cell_path, "X.csv"), DataFrame(X, :auto); writeheader = false)
    CSV.write(
        joinpath(cell_path, "y.csv"),
        DataFrame(y_pm = y_pm, y01 = y01);
        writeheader = true,
    )

    meta = Dict(
        "cell_id" => cell_id,
        "n" => N,
        "p" => P,
        "s" => S,
        "rho" => ρ_,
        "mu0" => μ0,
        "reg" => reg,
        "seed" => seed,
        "lambda" => lambda,
        "lambda_max" => lambda_max,
        "ybar" => ybar,
        "n_pos" => sum(y01 .== 1),
        "n_neg" => sum(y01 .== 0),
    )
    open(joinpath(cell_path, "meta.json"), "w") do io
        JSON3.pretty(io, meta)
    end

    push!(
        index_rows,
        (
            cell_id = cell_id,
            cell_name = cell_name,
            mu0 = μ0,
            reg = reg,
            seed = seed,
            n = N,
            p = P,
            lambda = lambda,
            lambda_max = lambda_max,
            ybar = ybar,
            n_pos = sum(y01 .== 1),
            n_neg = sum(y01 .== 0),
        ),
    )

    println(
        "[$cell_id/$(length(μ0_grid) * length(reg_grid) * length(seed_grid))] " *
            "$cell_name μ0=$μ0 reg=$reg seed=$seed " *
            "ybar=$(round(ybar; digits = 3)) λ=$(round(lambda; sigdigits = 4))",
    )
end

index_df = DataFrame(index_rows)
CSV.write(joinpath(outdir, "index.csv"), index_df)
println("Wrote $(nrow(index_df)) cells; index at $(joinpath(outdir, "index.csv"))")
