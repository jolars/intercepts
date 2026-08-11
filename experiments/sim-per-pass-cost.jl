using Intercepts
using Random
using DrWatson
using ProjectRoot
using JLD2

# Per-pass cost decomposition for the exact strategy.
# Records, per outer pass, the number of inner Newton steps consumed at the
# intercept (k_0) and the outer progress (relative gap), for both the Newton
# strategy (single inner step per pass) and the exact strategy (iterates
# until |∂_0 F| < 1e-10).
#
# Reads as @lem-exact-cost: Newton and exact achieve comparable
# per-pass progress, but the exact strategy pays Ω(k_0 - 1) extra inner steps
# per pass with no commensurate rate improvement.

Random.seed!(1234)

param_dict = Dict{String, Any}(
    "n" => [500],
    "p" => [1000],
    "s" => [10],
    "reg" => [0.05],
    "μ0" => [0.5, 0.9, 0.99],
    "strategy" => [:newton, :exact],
)

params = dict_list(param_dict)

const MAXIT = 200
const TOL = 1.0e-8

results = Vector{Dict{String, Any}}()

for (i, d) in enumerate(params)
    @unpack n, p, s, reg, μ0, strategy = d

    Random.seed!(2025)
    X, y = generatedata(
        n,
        p;
        response = :binomial,
        μ0 = μ0,
        x_type = :normal,
        ρ = 0.6,
        s = s,
        amplitude = 1.0,
    )

    intercept_strategy = if strategy == :newton
        NewtonStrategy()
    elseif strategy == :exact
        ExactStrategy()
    else
        error("Unknown strategy: $strategy")
    end

    res = cdsolver(
        X,
        Float64.(y),
        reg;
        lossfun = LogisticLoss(),
        intercept_strategy = intercept_strategy,
        randomize = false,
        tol = TOL,
        maxit = MAXIT,
    )

    # cdsolver records relgaps[t] before pass t starts. inner_steps[t] is the
    # number of inner intercept steps consumed during pass t. If the tolerance
    # halts the loop, relgaps has one more entry than inner_steps (the entry
    # logged at the halt). We report the relgap AT THE END of pass t — that is
    # relgaps[t+1] when available, falling back to relgaps[end].
    inner_steps = res.inner_steps
    T = length(inner_steps)
    n_rg = length(res.relgaps)
    relgap_after = Vector{Float64}(undef, T)
    for t in 1:T
        idx = min(t + 1, n_rg)
        relgap_after[t] = max(res.relgaps[idx], 1.0e-20)
    end
    pass = collect(1:T)

    d_exp = Dict{String, Any}(
        "n" => n,
        "p" => p,
        "s" => s,
        "reg" => reg,
        "μ0" => μ0,
        "strategy" => String(strategy),
        "pass" => pass,
        "relgap_after" => relgap_after,
        "inner_steps" => inner_steps,
        "cum_inner_steps" => cumsum(inner_steps),
        "time" => res.time[1:T],
        "passes" => res.passes,
    )

    println(
        "[$i/$(length(params))] μ0=$μ0 strategy=$strategy passes=$(res.passes) " *
            "total_inner_steps=$(sum(inner_steps)) final_relgap=$(relgap_after[end])",
    )

    push!(results, d_exp)
end

outfile = @projectroot("results", "sim-per-pass-cost.jld2")
@save outfile results
println("Saved results to $outfile")
