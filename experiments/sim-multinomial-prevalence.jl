using Intercepts
using Random
using ProjectRoot
using JLD2
using CSV
using DataFrames

# Reconstruct the seeded inputs without rerunning the cached optimization paths.
# The baseline experiment is also the (0.05, 0.5) corner of the sweep.
rows = NamedTuple[]
seen = Set{Tuple{Float64, Float64, Int}}()
for cache in ("sim-multinomial-imbalance.jld2", "sim-multinomial-imbalance-sweep.jld2")
    for record in JLD2.load(@projectroot("results", cache), "results")
        p_K = Float64(get(record, "p_K", 0.05))
        amplitude = Float64(get(record, "amplitude", 0.5))
        seed = record["it"]
        key = (p_K, amplitude, seed)
        key in seen && continue
        push!(seen, key)

        n, p, s, K = (record[name] for name in ("n", "p", "s", "K"))
        K == 5 || error("the cached experiment must have five classes")
        baseline_probs = [0.7 + (0.05 - p_K), 0.1, 0.1, 0.05, p_K]
        Random.seed!(seed)
        _, y = generatedata(
            n,
            p;
            response = :multinomial,
            K = K,
            class_probs = baseline_probs,
            s = s,
            ρ = 0.3,
            amplitude = amplitude,
            means = :random,
        )
        counts = [count(==(k), y) for k in 1:K]
        push!(
            rows,
            (
                baseline_p_K = p_K,
                amplitude = amplitude,
                seed = seed,
                n = n,
                n_class_1 = counts[1],
                n_class_2 = counts[2],
                n_class_3 = counts[3],
                n_class_4 = counts[4],
                n_class_5 = counts[5],
                all_classes_present = all(>(0), counts),
            ),
        )
    end
end

df = sort!(DataFrame(rows), [:baseline_p_K, :amplitude, :seed])
outfile = @projectroot("results", "sim-multinomial-prevalence.csv")
CSV.write(outfile, df)
println("Wrote $(nrow(df)) seeded class-count records to $outfile")
