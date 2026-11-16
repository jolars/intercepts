# Does striding the cheap gradient intercept close the gap to Newton?
#
# Section 3.3.3 says the gradient strategy pays a factor L0/H00 in passes
# because its 1/L0 step removes only the fraction H00/L0 of the coupling that
# each coefficient update injects. Striding attacks the same quantity from the
# other side: more intercept updates per pass, each still partial. If the two
# cancel, gradient-with-stride should land on Newton-at-baseline, and the
# cheap strategy stops costing anything.
#
# Paired within seed throughout; the k = p column is the baseline both
# strategies are measured against.

using DrWatson
@quickactivate "Intercepts"
using DataFrames, JLD2, Statistics, ProjectRoot, CSV

@load @projectroot("results", "sim-stride-logistic.jld2") df
df = df[df.converged, :]

CELL = [:n, :p, :μ0]
KEY = [CELL; :seed]

"Best wall time over strides, per seed, for one strategy."
function best_by_seed(d, strat)
    s = d[d.strategy .== strat, :]
    combine(groupby(s, KEY), :wall_time => minimum => :best_wall,
        :passes => minimum => :best_passes)
end

"The k = p baseline, per seed, for one strategy."
function base_by_seed(d, strat)
    s = d[(d.strategy .== strat) .& (d.ratio .== 1), :]
    rename(select(s, [KEY; :wall_time; :passes]),
        :wall_time => :base_wall, :passes => :base_passes)
end

g_best, n_base = best_by_seed(df, "gradient"), base_by_seed(df, "newton")
n_best, g_base = best_by_seed(df, "newton"), base_by_seed(df, "gradient")

# Ratio > 1 means the strided gradient beat the Newton baseline.
gap = innerjoin(g_best, n_base; on = KEY)
gap.gap_wall = gap.base_wall ./ gap.best_wall
gap.gap_passes = gap.base_passes ./ gap.best_passes

# The gap the stride had to close in the first place, same seeds.
loss = innerjoin(rename(g_base, :base_wall => :g_wall, :base_passes => :g_passes),
    n_base; on = KEY)
loss.loss_wall = loss.g_wall ./ loss.base_wall
loss.loss_passes = loss.g_passes ./ loss.base_passes

# How much of Newton's own advantage is left once Newton strides too.
nn = innerjoin(n_best, n_base; on = KEY)
nn.newton_gain = nn.base_wall ./ nn.best_wall

j = innerjoin(gap, select(loss, [KEY; :loss_wall; :loss_passes]),
    select(nn, [KEY; :newton_gain]); on = KEY, makeunique = true)

summ = sort(combine(groupby(j, CELL),
        :loss_wall => median => :loss,
        :gap_wall => median => :gap, :gap_wall => minimum => :gap_lo,
        :gap_wall => maximum => :gap_hi,
        :gap_wall => (v -> count(>=(0.95), v)) => :n_matched,
        :newton_gain => median => :newton_gain,
        nrow => :seeds), [:μ0, :p, :n])

println("\n=== Strided gradient vs Newton baseline (paired, wall time) ===")
println("loss      = gradient/Newton cost at the shared k = p baseline")
println("gap       = Newton baseline / best strided gradient  (1.0 = parity)")
println("matched   = seeds where the strided gradient came within 5% of Newton")
println("n_gain    = what Newton itself still gains from striding\n")
for r in eachrow(summ)
    println("  μ₀=", rpad(r.μ0, 6), " n=", rpad(Int(r.n), 6), " p=", rpad(Int(r.p), 6),
        "  loss=", lpad(round(r.loss; digits = 2), 5), "x",
        "  gap=", lpad(round(r.gap; digits = 2), 5), "x",
        " [", lpad(round(r.gap_lo; digits = 2), 4), "-", lpad(round(r.gap_hi; digits = 2), 4), "]",
        "  matched=", lpad(r.n_matched, 2), "/", r.seeds,
        "  n_gain=", lpad(round(r.newton_gain; digits = 2), 5), "x")
end

@info "overall" median_loss = median(j.loss_wall) median_gap = median(j.gap_wall) matched =
    count(>=(0.95), j.gap_wall) total = nrow(j)

CSV_OUT = @projectroot("results", "sim-stride-gap.csv")
CSV.write(CSV_OUT, summ)
@info "wrote" CSV_OUT

# The per-seed argmin above is chosen on the same run it is scored on, so it
# flatters the gradient strategy. Repeat with one stride fixed a priori for
# every cell, which is what a user would actually set.
for fixed in (10, 25, 50)
    gf = df[(df.strategy .== "gradient") .& (df.ratio .== fixed), :]
    gf = rename(select(gf, [KEY; :wall_time]), :wall_time => :fixed_wall)
    h = innerjoin(gf, n_base; on = KEY)
    h.gap = h.base_wall ./ h.fixed_wall
    sm = sort(combine(groupby(h, CELL), :gap => median => :gap,
            :gap => (v -> count(>=(0.95), v)) => :matched, nrow => :seeds),
        [:μ0, :p, :n])
    println("\n=== Fixed stride p/k = ", fixed, " vs Newton baseline ===")
    for r in eachrow(sm)
        println("  μ₀=", rpad(r.μ0, 6), " n=", rpad(Int(r.n), 6), " p=", rpad(Int(r.p), 6),
            "  gap=", lpad(round(r.gap; digits = 2), 5), "x  matched=",
            lpad(r.matched, 2), "/", r.seeds)
    end
    @info "fixed stride" fixed median_gap = median(h.gap) matched = count(>=(0.95), h.gap) total = nrow(h)
end
