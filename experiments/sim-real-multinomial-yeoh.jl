## Real-data multinomial complement to sim-multinomial-imbalance.jl: the
## Yeoh2002 (Yeoh, St. Jude, 2002) pediatric ALL gene-expression panel.
## n = 248 samples, p = 12 625 genes, K = 6 subtypes with class marginals
## (BCR=6%, MLL=8%, E2A=11%, T=17%, Hyperdip=26%, TEL=32%). The rare classes
## BCR and MLL sit in the regime where the gradient strategy's per-class
## fraction H_{00,kk}/L_0 is smallest, so the synthetic-imbalance story of
## @fig-multinomial-imbalance has a direct empirical analogue here.
##
## Fetch the data first via `Rscript experiments/fetch-yeoh.R`.

using Intercepts
using Random
using DrWatson
using ProjectRoot
using DataFrames
using CSV
using JLD2
using Statistics

Random.seed!(1234)

data_dir = @projectroot("data", "yeoh")
x_path = joinpath(data_dir, "X.csv.gz")
y_path = joinpath(data_dir, "y.csv")

if !isfile(x_path) || !isfile(y_path)
    error(
        "Yeoh data missing under $(data_dir). Run `Rscript experiments/fetch-yeoh.R` first.",
    )
end

X = Matrix(CSV.read(x_path, DataFrame; header = false, types = Float64))
y = vec(Matrix(CSV.read(y_path, DataFrame; header = false, types = Int)))

n, p = size(X)
K = length(unique(y))

println("Loaded Yeoh: n=$n, p=$p, K=$K")
for k = 1:K
    println("  class $k: ", count(==(k), y), " (", round(100 * mean(y .== k), digits = 2), "%)")
end

param_dict = Dict{String,Any}(
    "reg" => [0.05],
    "strategy" => [:gradient, :newton, :exact],
)

params = dict_list(param_dict)

results = []

for (i, d) in enumerate(params)
    @unpack reg, strategy = d
    println("Running strategy=$strategy ...")

    intercept_strategy = if strategy == :gradient
        GradientStrategy()
    elseif strategy == :newton
        NewtonStrategy()
    elseif strategy == :exact
        ExactStrategy()
    else
        error("Unknown strategy: $strategy")
    end

    lossfun = MultinomialLogisticLoss(K = K, lipschitz = 0.5)

    res = multinomial_cdsolver(
        X,
        y,
        reg;
        lossfun = lossfun,
        intercept_strategy = intercept_strategy,
        maxit = 1000,
        randomize = false,
    )

    d_exp = Dict{String,Any}(copy(d))
    d_exp["time"] = res.time
    d_exp["primals"] = res.primals
    d_exp["relgaps"] = max.(res.relgaps, 1e-20)
    d_exp["gaps"] = res.gaps

    push!(results, d_exp)
end

# Relative primal suboptimality against a shared optimum F* (best primal across
# strategies), certified by the duality gap. One problem instance, so all runs
# share a single F*.
suboptimality_against_certified_optimum!(results; instance_of = r -> 0)

outfile = @projectroot("results", "real-multinomial-yeoh.jld2")

@save outfile results
