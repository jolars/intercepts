# Does the simulated stride result hold on real data?
#
# sim-stride-logistic.jl predicted two things. First, that the 1/L0 intercept's
# handicap against the Armijo-guarded Newton intercept at the shared k = p
# baseline grows as the response becomes imbalanced. Second, that striding the
# 1/L0 intercept erases that handicap, landing it on Newton parity. This checks
# both on the shipped LIBSVM datasets, which were never used to tune anything
# here, and overlays them on the simulated points.

using DrWatson
@quickactivate "intercepts"
using DataFrames, JLD2, Statistics, Printf, ProjectRoot, CairoMakie, CSV

@load @projectroot("results", "sim-stride-real.jld2") df
real = df
@load @projectroot("results", "sim-stride-logistic.jld2") df
sim = df[df.converged, :]

n_censored = count(!, real.converged)
n_censored > 0 && @warn "censored real-data runs dropped" n_censored
real = real[real.converged, :]

# Pass counts are deterministic here, so they need no averaging; wall time does.
cell = combine(
    groupby(real, [:dataset, :n, :p, :pi_plus, :strategy, :k, :ratio]),
    :wall_time => median => :wall, :passes => first => :passes,
    :h00_l0 => first => :h00_l0,
)

base(d, s) = only(d[(d.strategy .== s) .& (d.ratio .== 1.0), :].wall)
basep(d, s) = only(d[(d.strategy .== s) .& (d.ratio .== 1.0), :].passes)

println("\n=== Real data: baseline handicap and what striding recovers ===")
println("handicap  = gradient / Newton at the shared k = p baseline")
println("best p/k  = stride minimising gradient wall time")
println("remaining = strided gradient / Newton baseline  (1.0 = parity)\n")

rows = NamedTuple[]
for g in groupby(sort(cell, :pi_plus), :dataset)
    ds = first(g.dataset)
    gr = g[g.strategy .== "gradient", :]
    gb, nb = base(g, "gradient"), base(g, "newton")
    gbp, nbp = basep(g, "gradient"), basep(g, "newton")
    i = argmin(gr.wall)
    best, bestp, bestr = gr.wall[i], gr.passes[i], gr.ratio[i]

    push!(
        rows,
        (
            dataset = ds, n = first(g.n), p = first(g.p),
            pi_plus = first(g.pi_plus), h00_l0 = first(g.h00_l0),
            handicap = gb / nb, handicap_passes = gbp / nbp,
            best_ratio = bestr, remaining = best / nb,
            remaining_passes = bestp / nbp, speedup = gb / best,
        ),
    )
    @printf(
        "  %-14s pi+=%.3f  H00/L0=%.3f  handicap=%5.2fx (%4d/%4d passes)  best p/k=%5.1f  remaining=%5.2fx  speedup=%5.2fx\n",
        ds, first(g.pi_plus), first(g.h00_l0), gb / nb, gbp, nbp, bestr, best / nb, gb / best
    )
end
summ = DataFrame(rows)
CSV.write(@projectroot("results", "sim-stride-real-summary.csv"), summ)

# --- the simulated points, computed the same way, for the overlay ----------
KEY = [:n, :p, :μ0, :seed]
col(d, nm) = rename(select(d, [KEY; :wall_time]), :wall_time => nm)
sb(s, r) = sim[(sim.strategy .== s) .& (sim.ratio .== r), :]
j = innerjoin(
    col(sb("gradient", 1), :g_base), col(sb("newton", 1), :n_base),
    col(sb("gradient", 25), :g_str); on = KEY,
)
j.handicap = j.g_base ./ j.n_base
j.remaining = j.g_str ./ j.n_base
simpts = combine(
    groupby(j, [:n, :p, :μ0]),
    :handicap => median => :handicap, :remaining => median => :remaining,
)

fig = Figure(size = (660, 480))
ax = Axis(
    fig[1, 1], xscale = log10, yscale = log10,
    xlabel = "Gradient / Newton wall time at the shared baseline (k = p)",
    ylabel = "Gradient / Newton wall time once strided",
    title = "Real data follows the simulated prediction",
    xticks = ([1, 2, 3, 5, 7], ["1", "2", "3", "5", "7"]),
    yticks = ([0.6, 0.8, 1.0, 1.2, 1.5, 2, 3, 5, 7],
        ["0.6", "0.8", "1.0", "1.2", "1.5", "2", "3", "5", "7"]),
)
hlines!(ax, [1.0]; color = :gray40, linestyle = :dash)
lines!(ax, [0.85, 7.5], [0.85, 7.5]; color = :gray70, linestyle = :dot)
text!(ax, 3.0, 3.6; text = "no gain from striding", color = :gray60,
    fontsize = 11, rotation = pi / 4)

scatter!(ax, simpts.handicap, simpts.remaining; color = (:gray50, 0.55),
    markersize = 11, label = "simulated (12 designs)")
scatter!(ax, summ.handicap, summ.remaining; color = :firebrick, marker = :diamond,
    markersize = 17, label = "real data")
for r in eachrow(summ)
    text!(ax, r.handicap, r.remaining; text = "  " * r.dataset, align = (:left, :center),
        fontsize = 11, color = :firebrick)
end
axislegend(ax; position = :lt, framevisible = false)

OUT = @projectroot("results", "sim-stride-real.png")
save(OUT, fig)
@info "wrote" OUT
