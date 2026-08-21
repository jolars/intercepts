## Real-data multinomial complement to sim-multinomial-imbalance.jl and the
## Yeoh2002 panel (sim-real-multinomial-yeoh.jl): the StatLog Shuttle
## dataset, K = 7 with class counts spanning four orders of magnitude
## (rarest class 0.014%, most common 78.4%). The (K, n, p) regime is
## complementary to Yeoh -- p = 9 sensor features at n = 58000 rather than
## p >> n gene expression -- so the rare-class universality claim is
## checked across two opposite high-dimensional regimes.

using Intercepts
using Random
using DrWatson
using ProjectRoot
using DataFrames
using JLD2
using Statistics

Random.seed!(1234)

# Combine the train and test splits; we are profiling optimization speed, not
# generalization, so the held-out split is not needed.
X_tr_raw, y_tr_raw = load_local_dataset("shuttle.scale")
X_te_raw, y_te_raw = load_local_dataset("shuttle.scale.t")

X = vcat(Matrix(X_tr_raw), Matrix(X_te_raw))
y = vcat(Int.(y_tr_raw), Int.(y_te_raw))

n, p = size(X)
K = length(unique(y))
@assert sort(unique(y)) == collect(1:K)

println("Loaded shuttle: n=$n, p=$p, K=$K")
for k in 1:K
    println(
        "  class $k: ",
        count(==(k), y),
        " (",
        round(100 * mean(y .== k), digits = 3),
        "%)",
    )
end

param_dict = Dict{String, Any}("reg" => [0.05], "strategy" => [:gradient, :newton, :exact])

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
        ExactStrategy(gradient_tol = 1.0e-8)
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

    d_exp = Dict{String, Any}(copy(d))
    d_exp["time"] = res.time
    d_exp["primals"] = res.primals
    d_exp["duals"] = res.duals
    d_exp["relgaps"] = max.(res.relgaps, 1.0e-20)
    d_exp["gaps"] = res.gaps

    push!(results, d_exp)
end

# Shared suboptimality bounds from the strongest feasible dual point. One
# problem instance means that every run shares the same bound.
suboptimality_against_shared_dual!(results; instance_of = r -> 0)

outfile = @projectroot("results", "real-multinomial-shuttle.jld2")

@save outfile results
