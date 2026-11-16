using JLD2
using DataFrames
using Statistics
using CairoMakie
using ProjectRoot

# Figure for the intercept update-stride sweep: median over seeds of passes and
# wall time against the number of intercept updates per pass (p/k). The band is
# the min--max spread across seeds.

@load @projectroot("results", "sim-update-frequency.jld2") results

df = DataFrame(
    family = [r["family"] for r in results],
    regime = [r["regime"] for r in results],
    μ0 = [r["μ0"] for r in results],
    strategy = [r["strategy"] for r in results],
    k = [r["k"] for r in results],
    upp = [r["updates_per_pass"] for r in results],
    seed = [r["seed"] for r in results],
    passes = [r["passes"] for r in results],
    wall = [r["wall_time"] for r in results],
    final_dual_relgap = [r["final_dual_relgap"] for r in results],
)

# Shared-dual rescoring can improve the gap of a capped run, so use the
# run's own duality gap to decide whether it reached the solver tolerance.
df.converged = df.final_dual_relgap .<= 1.0e-10
n_censored = count(!, df.converged)
n_censored > 0 && @warn "dropping capped runs from the stride ranking" n_censored total = nrow(df)
df = df[df.converged, :]
isempty(df) && error("No runs reached the relative-gap tolerance of 1e-10.")

function panel_label(family, regime, μ0)
    family == "normal" && return "Gaussian"
    family == "binomial" && return "Logistic  μ₀ = $(μ0)"
    family == "poisson" && return "Poisson  μ₀ = $(Int(μ0))"
    return "Multinomial  $(regime)"
end

df.panel = panel_label.(df.family, df.regime, df.μ0)

panel_order = [
    "Gaussian",
    "Logistic  μ₀ = 0.5", "Logistic  μ₀ = 0.9", "Logistic  μ₀ = 0.99",
    "Poisson  μ₀ = 10", "Poisson  μ₀ = 100",
    "Multinomial  balanced", "Multinomial  imbalanced",
]
panel_order = filter(in(Set(df.panel)), panel_order)

agg = combine(
    groupby(df, [:panel, :strategy, :upp, :k]),
    :passes => median => :passes_med,
    :passes => minimum => :passes_lo,
    :passes => maximum => :passes_hi,
    :wall => median => :wall_med,
    :wall => minimum => :wall_lo,
    :wall => maximum => :wall_hi,
    nrow => :n_seeds,
)

# Log axes default to tick labels like 10^1.64; place 1-2-5 decade ticks inside
# the data range and label them as plain integers instead.
function nice_ticks(lo, hi; decades_only = false)
    lo = max(lo, eps())
    mults = decades_only ? (1,) : (1, 2, 5)
    cand = Float64[]
    for e in floor(Int, log10(lo)):ceil(Int, log10(hi)), m in mults
        v = m * 10.0^e
        lo / 1.05 <= v <= hi * 1.05 && push!(cand, v)
    end
    length(cand) < 2 && (cand = [lo, hi])
    return sort(cand)
end

fmt_num(v) = v >= 1 ? string(round(Int, v)) : string(round(v; sigdigits = 2))
fmt_nums(vals) = [fmt_num(v) for v in vals]

strategy_style = [
    ("gradient", "Gradient (1/L₀)", :steelblue),
    ("newton", "Newton", :darkorange),
    ("exact", "Exact", :seagreen),
]

set_theme!(theme_latexfonts(); fontsize = 13)
fig = Figure(; size = (1900, 680))

rows = [
    (:passes_med, :passes_lo, :passes_hi, "Passes to relative gap ≤ 1e-10"),
    (:wall_med, :wall_lo, :wall_hi, "Wall time (s, median over seeds)"),
]

for (ri, (medcol, locol, hicol, ylabel)) in enumerate(rows), (ci, pan) in enumerate(panel_order)
    sub_panel = agg[agg.panel .== pan, :]
    isempty(sub_panel) && continue

    xt = nice_ticks(minimum(sub_panel.upp), maximum(sub_panel.upp); decades_only = true)
    ylo = minimum(sub_panel[!, locol])
    yhi = maximum(sub_panel[!, hicol])
    yt = nice_ticks(ylo, yhi)

    ax = Axis(
        fig[ri, ci];
        title = ri == 1 ? pan : "",
        ylabel = ci == 1 ? ylabel : "",
        xscale = log10,
        yscale = log10,
        xticks = (xt, fmt_nums(xt)),
        yticks = (yt, fmt_nums(yt)),
        xgridvisible = false,
        ygridvisible = false,
    )

    for (strat, _, color) in strategy_style
        sub = sort(sub_panel[sub_panel.strategy .== strat, :], :upp)
        isempty(sub) && continue
        band!(ax, sub.upp, sub[!, locol], sub[!, hicol]; color = (color, 0.12))
        lines!(ax, sub.upp, sub[!, medcol]; color = color, linewidth = 2.5)
        scatter!(ax, sub.upp, sub[!, medcol]; color = color, markersize = 7)
    end

    # k = p is the paper's baseline: one update per pass, at its end.
    vlines!(ax, [1.0]; color = :gray60, linestyle = :dash, linewidth = 1)
end

Label(
    fig[length(rows) + 1, 1:length(panel_order)],
    "Intercept updates per pass   (p / k)";
    fontsize = 14,
    padding = (0, 0, 2, 0),
)

Legend(
    fig[1:length(rows), length(panel_order) + 1],
    [LineElement(; color = c, linewidth = 2) for (_, _, c) in strategy_style],
    [lab for (_, lab, _) in strategy_style],
    "Intercept strategy";
    framevisible = false,
)

Label(
    fig[0, 1:(length(panel_order) + 1)],
    "Intercept update stride: dashed line is k = p, the once-per-pass baseline; " *
        "line is the median over up to $(maximum(agg.n_seeds)) converged seeds, band the min-max across them";
    fontsize = 15,
    padding = (0, 0, 6, 0),
)

outfile = @projectroot("results", "sim-update-frequency.png")
save(outfile, fig; px_per_unit = 2)
println("Saved figure to $outfile")

# Companion table: best stride per panel and strategy, by median wall time.
best = combine(groupby(agg, [:panel, :strategy])) do g
    i = argmin(g.wall_med)
    base = g[g.upp .== 1, :]
    (
        best_k = g.k[i], best_upp = g.upp[i],
        best_passes = g.passes_med[i], best_wall = round(g.wall_med[i], digits = 3),
        best_n_seeds = g.n_seeds[i],
        base_passes = isempty(base) ? missing : base.passes_med[1],
        base_wall = isempty(base) ? missing : round(base.wall_med[1], digits = 3),
        base_n_seeds = isempty(base) ? 0 : base.n_seeds[1],
        wall_speedup = isempty(base) ? missing :
            round(base.wall_med[1] / g.wall_med[i], digits = 2),
    )
end
sort!(best, [:panel, :strategy])
println(best)
CSV_OUT = @projectroot("results", "sim-update-frequency-best.csv")
using CSV
CSV.write(CSV_OUT, best)
println("Saved table to $CSV_OUT")
