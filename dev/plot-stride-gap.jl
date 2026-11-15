# One panel: the gradient strategy's handicap at the shared k = p baseline
# against what is left of it once the gradient intercept fires every p/25
# coordinates. Points far to the right started out 5-6x slower than Newton;
# if striding closes the gap they still land on the parity line.

using DrWatson
@quickactivate "intercepts"
using DataFrames, JLD2, Statistics, ProjectRoot, CairoMakie

@load @projectroot("results", "sim-stride-logistic.jld2") df
df = df[df.converged, :]

KEY = [:n, :p, :μ0, :seed]
CELL = [:n, :p, :μ0]
FIXED = 25

sub(strat, ratio) = df[(df.strategy .== strat) .& (df.ratio .== ratio), :]
col(d, name) = rename(select(d, [KEY; :wall_time]), :wall_time => name)

j = innerjoin(col(sub("gradient", 1), :g_base), col(sub("newton", 1), :n_base),
    col(sub("gradient", FIXED), :g_strided); on = KEY)
j.handicap = j.g_base ./ j.n_base          # how much the cheap step costs at k = p
j.remaining = j.g_strided ./ j.n_base      # how much it costs once strided

s = sort(combine(groupby(j, CELL),
        :handicap => median => :handicap,
        :remaining => median => :remaining,
        :remaining => minimum => :lo, :remaining => maximum => :hi), :μ0)

MU = sort(unique(s.μ0))
colors = [:steelblue, :darkorange, :firebrick]

fig = Figure(size = (620, 460))
ax = Axis(fig[1, 1], xscale = log10, yscale = log10,
    xlabel = "Gradient / Newton wall time at the shared baseline (k = p)",
    ylabel = "Gradient / Newton wall time at stride p/k = $FIXED",
    title = "Striding the 1/L₀ intercept erases the gap to Newton",
    xticks = ([1, 2, 3, 5, 7], ["1", "2", "3", "5", "7"]),
    yticks = ([0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2, 3, 5, 7],
        ["0.6", "0.7", "0.8", "0.9", "1.0", "1.2", "1.5", "2", "3", "5", "7"]))

hlines!(ax, [1.0]; color = :gray40, linestyle = :dash)
lines!(ax, [0.9, 7.0], [0.9, 7.0]; color = :gray70, linestyle = :dot)
text!(ax, 3.2, 4.0; text = "no gain from striding", color = :gray60, fontsize = 11,
    rotation = pi / 4)
text!(ax, 2.6, 1.04; text = "parity with Newton", color = :gray40, fontsize = 11)

for (i, mu) in enumerate(MU)
    d = s[s.μ0 .== mu, :]
    rangebars!(ax, d.handicap, d.lo, d.hi; color = colors[i], whiskerwidth = 8)
    scatter!(ax, d.handicap, d.remaining; color = colors[i], markersize = 13,
        label = "μ₀ = $mu")
end
axislegend(ax, "Response mean", position = :lt, framevisible = false)

OUT = @projectroot("results", "sim-stride-gap.png")
save(OUT, fig)
@info "wrote" OUT median_remaining = median(j.remaining) max_handicap = maximum(s.handicap)
