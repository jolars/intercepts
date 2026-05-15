using Intercepts
using Random
using GLM
using DrWatson
using ProjectRoot
using DataFrames
using JLD2
using Statistics
using LinearAlgebra

Random.seed!(42);

# "news20-imb" is a subsampled news20.binary with controlled imbalance
# (3 % positive, 2 000 samples) standing in for the severely-imbalanced
# high-dimensional regime missing from the LIBSVM binary catalogue outside the
# Webb-pages w-series.
param_dict = Dict{String,Any}(
    "dataset" => ["w1a", "news20-imb", "a4a", "leukemia", "breast-cancer", "gisette_scale"],
    "reg" => [0.05],
    "strategy" => [:gradient, :newton, :exact],
);

params = dict_list(param_dict);

results = [];

for (i, d) in enumerate(params)
    @unpack dataset, reg, strategy = d

    if dataset == "news20-imb"
        # n = 2 000 with 3 % positive class matches w1a's $(n, \mu_0)$ scale;
        # drop features that appear in fewer than 1 % of the surviving rows to
        # avoid the standardisation-by-tiny-std numerical issue that
        # destabilises the CD solve on the full p = 1 355 191 vocabulary, and
        # cap at 120 s/strategy as a safety belt.
        res = real_experiment(
            "news20.binary";
            strategy = strategy,
            reg = reg,
            response = :binomial,
            randomize = false,
            n_positive = 60,
            n_negative = 1940,
            min_nnz_per_column = 20,
            seed = 42,
            maxtime = 120.0,
        )
    else
        res = real_experiment(
            dataset;
            strategy = strategy,
            reg = reg,
            response = :binomial,
            randomize = false,
        )
    end

    d_exp = Dict{String,Any}(copy(d))
    d_exp["time"] = res.time
    d_exp["gaps"] = res.gaps
    d_exp["relgaps"] = res.relgaps
    d_exp["primals"] = res.primals

    push!(results, d_exp)
end

outfile = @projectroot("results", "real-logreg.jld2");

@save outfile results

