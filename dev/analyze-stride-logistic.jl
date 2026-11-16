using JLD2
using DataFrames
using Statistics
using CairoMakie
using ProjectRoot
using CSV
using Printf

@load @projectroot("results", "sim-stride-logistic.jld2") df

KEY = [:n, :p, :μ0, :strategy, :seed]

# A censored run stopped at maxit/maxtime, so its wall time is a cap and its
# pass count a lower bound. Both would bias the paired ratios toward 1 and the
# Eq. 16 test downward, so drop them; the innerjoin below then also drops any
# seed whose own baseline is censored.
n_censored = count(!, df.converged)
if n_censored > 0
    @warn "dropping censored runs from the paired analysis" n_censored total = nrow(df)
end
df = df[df.converged, :]

# ---------------------------------------------------------------- paired data
# Every comparison is within a seed: instance variation is larger than the
# effect, so ratio-of-medians across seeds understates what the design shows.
base = rename(
    select(df[df.ratio .== 1, :], [KEY; :wall_time; :passes]),
    :wall_time => :base_wall, :passes => :base_passes,
)
d = innerjoin(df, base; on = KEY)
d.speedup = d.base_wall ./ d.wall_time
d.pass_speedup = d.base_passes ./ d.passes

summ = combine(
    groupby(d, [:n, :p, :np_ratio, :μ0, :strategy, :ratio]),
    :speedup => median => :speedup_med,
    :speedup => minimum => :speedup_min,
    :speedup => maximum => :speedup_max,
    :speedup => (v -> count(>(1.0), v)) => :n_faster,
    :speedup => length => :n_seeds,
    :wall_time => median => :wall_med,
    :passes => median => :passes_med,
    :h00_l0 => median => :h00_l0,
)

# --------------------------------------------------- per-seed best stride (Q1)
# The argmin of an averaged curve hides how stable the choice is, so take the
# argmin within each seed and report its distribution.
argmins = combine(groupby(d, KEY)) do g
    i = argmin(g.wall_time)
    (best_ratio = g.ratio[i], best_k = g.k[i], best_wall = g.wall_time[i])
end

argmin_summ = combine(
    groupby(argmins, [:n, :p, :μ0, :strategy]),
    :best_ratio => median => :med_best_ratio,
    :best_ratio => minimum => :min_best_ratio,
    :best_ratio => maximum => :max_best_ratio,
    :best_k => median => :med_best_k,
)

# ----------------------------------------------------- theory check (Eq. 16)
# T_G / T_N -> L0 / H00 as the coupling grows. Compare at the baseline stride,
# where the intercept mode is least masked by in-pass updates.
THEORY_KEY = [:n, :p, :μ0, :seed]
baseline = df[df.ratio .== 1, :]
# A theory ratio requires both strategies to converge on the same seed.
gn = innerjoin(
    rename(
        select(baseline[baseline.strategy .== "gradient", :], [THEORY_KEY; :passes; :h00_l0]),
        :passes => :gradient,
    ),
    rename(
        select(baseline[baseline.strategy .== "newton", :], [THEORY_KEY; :passes]),
        :passes => :newton,
    );
    on = THEORY_KEY,
)
gn.observed = gn.gradient ./ gn.newton
gn.predicted = 1.0 ./ gn.h00_l0

theory = if isempty(gn)
    DataFrame(
        n = Int[], p = Int[], μ0 = Float64[], obs_med = Float64[],
        obs_min = Float64[], obs_max = Float64[], pred_med = Float64[], h00_l0 = Float64[],
    )
else
    combine(
        groupby(gn, [:n, :p, :μ0]),
        :observed => median => :obs_med,
        :observed => minimum => :obs_min,
        :observed => maximum => :obs_max,
        :predicted => median => :pred_med,
        :h00_l0 => median => :h00_l0,
    )
end
theory.rel_error = (theory.obs_med .- theory.pred_med) ./ theory.pred_med

# ------------------------------------------------------------------- figure 1
set_theme!(theme_latexfonts(); fontsize = 13)

configs = sort(unique(select(df, [:n, :p, :np_ratio])), [:np_ratio, :p])
mus = sort(unique(df.μ0))
styles = [("gradient", "Gradient (1/L₀)", :steelblue), ("newton", "Newton", :darkorange)]

fig = Figure(; size = (1250, 1180))
for (ri, cfg) in enumerate(eachrow(configs)), (ci, μ) in enumerate(mus)
    sub_all = summ[(summ.n .== cfg.n) .& (summ.p .== cfg.p) .& (summ.μ0 .== μ), :]
    isempty(sub_all) && continue
    ax = Axis(
        fig[ri, ci];
        title = ri == 1 ? "μ₀ = $μ" : "",
        ylabel = ci == 1 ? "n = $(cfg.n), p = $(cfg.p)\n(n/p = $(cfg.np_ratio))  wall time (s)" : "",
        xlabel = ri == nrow(configs) ? "Inverse relative stride  (p / k)" : "",
        xscale = log10, yscale = log10, xgridvisible = false, ygridvisible = false,
    )
    for (strat, _, color) in styles
        s = sort(sub_all[sub_all.strategy .== strat, :], :ratio)
        isempty(s) && continue
        band!(ax, s.ratio, s.wall_med .* s.speedup_med ./ s.speedup_max,
              s.wall_med .* s.speedup_med ./ s.speedup_min; color = (color, 0.12))
        lines!(ax, s.ratio, s.wall_med; color = color, linewidth = 2.5)
        scatter!(ax, s.ratio, s.wall_med; color = color, markersize = 7)
        am = argmin_summ[(argmin_summ.n .== cfg.n) .& (argmin_summ.p .== cfg.p) .&
                         (argmin_summ.μ0 .== μ) .& (argmin_summ.strategy .== strat), :]
        isempty(am) || vlines!(ax, [am.med_best_ratio[1]]; color = color,
                               linestyle = :dot, linewidth = 1.5)
    end
end
Legend(
    fig[nrow(configs) + 1, 1:length(mus)],
    [LineElement(; color = c, linewidth = 2.5) for (_, _, c) in styles],
    [l for (_, l, _) in styles]; orientation = :horizontal, framevisible = false,
)
Label(
    fig[0, 1:length(mus)],
    "Logistic intercept stride: median wall time over $(maximum(summ.n_seeds)) seeds, " *
        "band = min-max of the paired per-seed speedup; dotted line = median per-seed best stride";
    fontsize = 14,
)
save(@projectroot("results", "sim-stride-logistic.png"), fig; px_per_unit = 2)

# ------------------------------------------------------- figure 2 (Eq. 16)
if isempty(theory)
    @warn "skipping theory plot: no seeds have both converged baselines"
else
    fig2 = Figure(; size = (560, 500))
    ax2 = Axis(
        fig2[1, 1];
        xlabel = "Predicted  L₀ / H₀₀  (measured at the fit)",
        ylabel = "Observed  gradient passes / Newton passes",
        xscale = log10, yscale = log10, xgridvisible = false, ygridvisible = false,
        title = "Eq. 16 against the sweep (stride k = p)",
    )
    lims = (0.9 * minimum(min.(theory.pred_med, theory.obs_med)),
            1.1 * maximum(max.(theory.pred_med, theory.obs_med)))
    lines!(ax2, [lims...], [lims...]; color = :gray60, linestyle = :dash)
    for (ci, μ) in enumerate(mus)
        t = theory[theory.μ0 .== μ, :]
        isempty(t) && continue
        scatter!(ax2, t.pred_med, t.obs_med; markersize = 13, label = "μ₀ = $μ")
        rangebars!(ax2, t.pred_med, t.obs_min, t.obs_max; whiskerwidth = 8)
    end
    axislegend(ax2; position = :lt, framevisible = false)
    save(@projectroot("results", "sim-stride-logistic-theory.png"), fig2; px_per_unit = 2)
end

# -------------------------------------------------------------------- tables
CSV.write(@projectroot("results", "sim-stride-logistic-summary.csv"), summ)
CSV.write(@projectroot("results", "sim-stride-logistic-argmin.csv"), argmin_summ)
CSV.write(@projectroot("results", "sim-stride-logistic-theory.csv"), theory)

println("\n=== Q1: per-seed best stride (p/k), median [min-max] over seeds ===")
for r in eachrow(sort(argmin_summ, [:strategy, :μ0, :p, :n]))
    @printf(
        "  %-8s μ₀=%-5s n=%-5d p=%-5d  best p/k = %6.2f [%6.2f-%6.2f]   best k = %5.0f\n",
        r.strategy, r.μ0, r.n, r.p, r.med_best_ratio,
        r.min_best_ratio, r.max_best_ratio, r.med_best_k
    )
end

println("\n=== Q2: Eq. 16, observed vs predicted gradient/Newton pass ratio ===")
for r in eachrow(sort(theory, [:μ0, :p, :n]))
    @printf(
        "  μ₀=%-5s n=%-5d p=%-5d  H₀₀/L₀=%.3f  predicted=%5.2f  observed=%5.2f [%4.2f-%5.2f]  err=%+5.1f%%\n",
        r.μ0, r.n, r.p, r.h00_l0, r.pred_med, r.obs_med, r.obs_min, r.obs_max, 100 * r.rel_error
    )
end

println("\n=== Paired speedup vs stride k=p, gradient, by cell (sign test) ===")
g = sort(summ[(summ.strategy .== "gradient"), :], [:μ0, :p, :n, :ratio])
for r in eachrow(g[g.ratio .∈ Ref([10, 20, 25, 50]), :])
    @printf(
        "  μ₀=%-5s n=%-5d p=%-5d p/k=%-6.2f  speedup %5.2fx [%4.2f-%5.2f]  %d/%d seeds faster\n",
        r.μ0, r.n, r.p, r.ratio, r.speedup_med, r.speedup_min, r.speedup_max,
        r.n_faster, r.n_seeds
    )
end
