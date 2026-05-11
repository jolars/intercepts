using Intercepts
using Random
using LinearAlgebra
using Statistics
using ProjectRoot
using CSV
using DataFrames
using JSON3

# Shared problem for the three production-solver experiments (glmnet, biglasso,
# skglm). Generates the imbalanced logistic problem of fig-irls-comparison and
# writes it to results/real-solvers/ in formats consumable by R and Python.
#
# Convention used downstream:
#   - y is stored as -1/+1
#   - lambda is on the "(1/n) * sum(loss) + lambda * ||beta||_1" scale
#     (glmnet / skglm / biglasso convention)
#   - lambda_max = max_j |X_j^T (y - ybar)| / n where ybar = mean(y in {0,1})
#   - lambda = reg * lambda_max with reg = 0.05

Random.seed!(2025)

n = 500
p = 1000
s = 10
μ0 = 0.99
reg = 0.05

X_raw, y01 = generatedata(
    n,
    p;
    response = :binomial,
    μ0 = μ0,
    x_type = :normal,
    ρ = 0.6,
    s = s,
    amplitude = 1.0,
)

# Pre-standardize X (column-wise center and scale to unit variance, biased SD)
# so that every solver receives the same matrix and the L1 penalty has a
# uniform interpretation. glmnet/skglm are given standardize=FALSE; biglasso's
# internal standardization is a no-op on the already-standardized input.
x_centers = vec(mean(X_raw; dims = 1))
x_scales = vec(std(X_raw; dims = 1, corrected = false))
x_scales[x_scales .== 0] .= 1.0
X = (X_raw .- x_centers') ./ x_scales'

y_pm = 2.0 .* y01 .- 1.0
ybar = mean(y01)

lambda_max = norm(X' * (y01 .- ybar), Inf) / n
lambda = reg * lambda_max

outdir = @projectroot("results", "real-solvers")
mkpath(outdir)

CSV.write(joinpath(outdir, "X.csv"), DataFrame(X, :auto); writeheader = false)
CSV.write(
    joinpath(outdir, "y.csv"),
    DataFrame(y_pm = y_pm, y01 = y01);
    writeheader = true,
)

meta = Dict(
    "n" => n,
    "p" => p,
    "s" => s,
    "mu0" => μ0,
    "reg" => reg,
    "lambda" => lambda,
    "lambda_max" => lambda_max,
    "ybar" => ybar,
    "n_pos" => sum(y01 .== 1),
    "n_neg" => sum(y01 .== 0),
    "seed" => 2025,
)

open(joinpath(outdir, "meta.json"), "w") do io
    JSON3.pretty(io, meta)
end

println("Wrote shared problem to $outdir")
println("  n_pos = $(meta["n_pos"]), n_neg = $(meta["n_neg"])")
println("  lambda_max = $lambda_max, lambda = $lambda")
