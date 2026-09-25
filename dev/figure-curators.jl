# Per-figure table curators for dev/figure.jl, keyed by chunk label.
#
# A curator takes the evaluated chunk's module and returns one DataFrame: the
# figure's information in tabular form, plus whatever derived quantity the
# surrounding prose actually argues from. Without a curator a chunk falls back
# to dumping the data frames it built, which is fine for small frames and noisy
# for large ones.
#
# Round aggressively. These tables are tracked, so their `git diff` after a
# re-run is meant to show which figures genuinely moved; unrounded floats turn
# last-digit wobble into diff noise and defeat the point.

const CURATORS = Dict{String, Function}(

    # The empirical-to-limit ratio shows where the pass counts approach the
    # strong-coupling plateau; the curvature fraction checks that centering
    # preserves the intercept curvature.
    "fig-rho-centering" => function (mod)
        df = sort(mod.df, [:μ0, :barρ2])
        return DataFrame(
            μ0 = df.μ0,
            α = round.(df.α; digits = 2),
            barρ2 = round.(df.barρ2; digits = 4),
            empirical = round.(df.empirical_ratio; digits = 3),
            asymptotic = round.(df.asymptotic_ratio; digits = 3),
            emp_over_asymptotic = round.(df.empirical_ratio ./ df.asymptotic_ratio; digits = 3),
            H00_over_L0 = round.(df.H00_over_L0; digits = 4),
            T_gradient = df.T_gradient,
            T_newton = df.T_newton,
        )
    end,

)
