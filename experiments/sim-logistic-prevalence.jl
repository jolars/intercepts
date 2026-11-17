using Intercepts
using CSV
using DataFrames
using JLD2
using ProjectRoot
using Random
using Statistics

# Recover realized response counts without running any optimization. The
# feature means matter: simulated_experiment uses :random, whereas the IRLS,
# per-pass-cost, warm-start, and controlled-skglm producers omit that option.
function main()
    rows = NamedTuple[]
    counts = Dict{Tuple, Int}()
    seen = Set{Tuple}()

    function record!(source, n, p, s, rho, mu0, seed, means)
        key = (n, p, s, rho, mu0, seed, means)
        source_key = (source, key)
        source_key in seen && return
        push!(seen, source_key)
        n_positive = get!(counts, key) do
            Random.seed!(seed)
            _, y = generatedata(
                n, p; response = :binomial, μ0 = mu0, s = s, ρ = rho,
                amplitude = 1.0, x_type = :normal, means = means,
            )
            count(==(1), y)
        end
        pi_plus = n_positive / n
        push!(
            rows, (
                experiment = source, seed = seed, n = n, p = p, s = s, rho = rho,
                mu0 = mu0, feature_means = means === nothing ? "zero" : string(means),
                n_positive = n_positive, n_negative = n - n_positive,
                pi_plus = pi_plus, intercept_only_fraction = 4pi_plus * (1 - pi_plus),
            )
        )
        return nothing
    end

    for (source, means) in (
            ("sim-mu-extreme", :random), ("sim-mu-reg-heatmap", :random),
            ("sim-cold-start", :random), ("sim-irls-comparison", nothing),
            ("sim-per-pass-cost", nothing),
        )
        records = JLD2.load(@projectroot("results", "$source.jld2"), "results")
        for r in records
            seed = get(r, "seed", get(r, "it", 2025))
            record!(source, r["n"], r["p"], r["s"], get(r, "ρ", 0.6), r["μ0"], seed, means)
        end
    end
    record!("sim-warmstart-path", 500, 1000, 10, 0.6, 0.9, 2025, nothing)

    # This index already stores response counts. Reuse them rather than
    # reconstructing the separate controlled-skglm problem family.
    for r in CSV.File(@projectroot("results", "skglm-controlled", "index.csv"))
        key = ("skglm-controlled", r.seed, r.mu0)
        key in seen && continue
        push!(seen, key)
        push!(
            rows, (
                experiment = "skglm-controlled", seed = r.seed, n = r.n, p = r.p,
                s = 10, rho = 0.6, mu0 = r.mu0, feature_means = "zero",
                n_positive = r.n_pos, n_negative = r.n_neg, pi_plus = r.ybar,
                intercept_only_fraction = 4r.ybar * (1 - r.ybar),
            )
        )
    end

    df = sort!(DataFrame(rows), [:experiment, :mu0, :seed])
    outfile = @projectroot("results", "logistic-prevalence.csv")
    CSV.write(outfile, df)
    println("Wrote $(nrow(df)) response-count records to $outfile; no solvers run.")
    return nothing
end

main()
