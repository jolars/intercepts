# Dev-facing figure inspector. Renders a single labeled chunk of intercepts.qmd
# to PNG and writes a summary table of the data behind it, without a full Quarto
# render and without competing with a live `quarto preview` for the notebook
# runner's Julia worker.
#
#   julia --project=. dev/figure.jl fig-rho-centering
#   julia --project=. dev/figure.jl --all
#   julia --project=. dev/figure.jl --full fig-real-glmnet   # uncapped table
#
# Outputs, per label:
#   results/figure-tables/<label>.csv       tracked --- diffing these after a
#                                           re-run shows which figures moved
#   results/figure-tables/png/<label>.png   gitignored (binary churn)
#
# The PNG exists because rendered figures are SVG, which is unreadable to a
# coding agent (glyph outline paths, no picture). The CSV exists because the
# plotting chunk transforms its inputs --- stack, relabel, sort, filter --- and
# bugs in that transform are invisible in the source .jld2.
#
# Table contents default to the data frames the chunk itself built, capped. A
# label listed in dev/figure-curators.jl overrides that with a hand-written
# summary; curation lives here rather than in intercepts.qmd so the paper source
# stays free of dev scaffolding.

using AlgebraOfGraphics
using CairoMakie
using DataFrames
using CSV

const QMD = joinpath(@__DIR__, "..", "intercepts.qmd")
const OUTDIR = joinpath(@__DIR__, "..", "results", "figure-tables")
const PNGDIR = joinpath(OUTDIR, "png")
const ROWCAP = 200

"""
    chunks(path)

Parse a Quarto notebook into `label => body` pairs, one per `{julia}` chunk.
Chunk options (the `#|` lines) are stripped; an unlabeled chunk is skipped.
"""
function chunks(path)
    out = Pair{String, String}[]
    label = nothing
    body = String[]
    inchunk = false
    for line in eachline(path)
        if !inchunk
            startswith(line, "```{julia}") && (inchunk = true; label = nothing; body = String[])
        elseif startswith(line, "```")
            inchunk = false
            label === nothing || push!(out, label => join(body, "\n"))
        elseif startswith(line, "#|")
            m = match(r"^#\|\s*label:\s*(\S+)", line)
            m === nothing || (label = m.captures[1])
        else
            push!(body, line)
        end
    end
    return out
end

"""
    eval_body!(mod, body)

Evaluate one chunk body in `mod`, returning its final expression's value (the
figure, for a plotting chunk) and the set of names that existed beforehand. The
`filename` keyword gives `@projectroot` a file context to resolve from.
"""
function eval_body!(mod, body)
    before = Set(Base.invokelatest(names, mod; all = true))
    value = Core.eval(mod, Meta.parseall(body; filename = abspath(QMD)))
    return value, before
end

"""
    run_chunk(all_chunks, label; cumulative = false)

Evaluate a labeled chunk in a fresh module. Quarto runs the notebook as a single
session, so some chunks deliberately reuse a predecessor's bindings ---
`fig-per-pass-progress` reuses the frame built by `fig-per-pass-cost`. Isolated
evaluation is much faster and works for most chunks, so we try that first and
fall back to replaying every chunk up to the target only when the isolated run
reports an undefined name.
"""
function run_chunk(all_chunks, label; cumulative = false)
    setup_body = last(all_chunks[findfirst(p -> first(p) == "setup", all_chunks)])
    target = findfirst(p -> first(p) == label, all_chunks)
    mod = Module(:Chunk)
    eval_body!(mod, setup_body)
    if cumulative
        for (l, body) in all_chunks[1:(target - 1)]
            l == "setup" && continue
            try
                eval_body!(mod, body)
            catch
                # A predecessor that fails on its own is not this chunk's
                # problem; keep going and let the target report if it needs it.
            end
        end
    end
    value, before = eval_body!(mod, last(all_chunks[target]))
    return mod, value, before
end

"""
    frames(mod, setup_names)

Data frames the chunk introduced, as `name => df`, excluding anything inherited
from the setup chunk. Ordered as defined, so the last entry is normally the one
handed to the plot.
"""
function frames(mod, setup_names)
    out = Pair{Symbol, DataFrame}[]
    for n in Base.invokelatest(names, mod; all = true)
        n in setup_names && continue
        # invokelatest throughout: the chunk's globals were created by Core.eval
        # in a newer world than this function was compiled in, and Julia 1.12
        # applies world age to bindings, not just methods. Without it `isdefined`
        # quietly reports false for every binding the chunk just made.
        Base.invokelatest(isdefined, mod, n) || continue
        v = Base.invokelatest(getfield, mod, n)
        v isa DataFrame && push!(out, n => v)
    end
    return out
end

"""
    capped(df, cap)

`df` if it fits, otherwise its head and tail with a marker row between, so a
5000-row trajectory frame stays readable without hiding that it was truncated.
"""
function capped(df, cap)
    nrow(df) <= cap && return df, false
    half = cap ÷ 2
    return vcat(first(df, half), last(df, half)), true
end

"""
    scalarize(df)

Replace collection-valued cells with a `Type(length)` descriptor. Several chunks
hold whole per-iteration trajectories in a single cell before `flatten`, and
writing those verbatim turns a 200-row table into megabytes of numbers that say
nothing about the figure.
"""
function scalarize(df)
    out = copy(df)
    for col in names(out)
        v = out[!, col]
        eltype(v) <: Union{Number, AbstractString, Symbol, Missing, Nothing, Bool} && continue
        out[!, col] = map(v) do x
            x isa AbstractArray ? "$(eltype(x))($(length(x)))" : string(x)
        end
    end
    return out
end

function write_tables(label, tables, full)
    paths = String[]
    for (name, df) in tables
        df = scalarize(df)
        out, was_capped = full ? (df, false) : capped(df, ROWCAP)
        suffix = length(tables) == 1 ? "" : "-$(name)"
        path = joinpath(OUTDIR, "$(label)$(suffix).csv")
        CSV.write(path, out)
        push!(paths, path)
        note = was_capped ? "  [capped from $(nrow(df)) rows; --full for all]" : ""
        println("  table $(basename(path))  $(nrow(out))x$(ncol(out))$(note)")
        println("    cols: ", join(names(df), ", "))
    end
    return paths
end

function inspect(label, all_chunks; full = false)
    mod, value, setup_names = try
        run_chunk(all_chunks, label)
    catch err
        err isa UndefVarError || rethrow()
        println("  (replaying earlier chunks: needs `$(err.var)` from a predecessor)")
        run_chunk(all_chunks, label; cumulative = true)
    end

    if value isa Union{AlgebraOfGraphics.FigureGrid, Makie.Figure}
        path = joinpath(PNGDIR, "$(label).png")
        save(path, value; px_per_unit = 2)
        println("  png   $(basename(path))")
    else
        println("  png   (skipped: chunk returned $(typeof(value)))")
    end

    curated = get(CURATORS, label, nothing)
    tables = if curated === nothing
        fs = frames(mod, setup_names)
        # A chunk that reuses a predecessor's frame builds none of its own; fall
        # back to whatever is in scope so it still gets a table.
        isempty(fs) && (fs = frames(mod, Set{Symbol}()))
        fs
    else
        [:summary => Base.invokelatest(curated, mod)]
    end
    isempty(tables) && println("  table (none: chunk defined no DataFrame)")
    write_tables(label, tables, full)
    return nothing
end

include(joinpath(@__DIR__, "figure-curators.jl"))

function main(args)
    full = "--full" in args
    wanted = filter(a -> !startswith(a, "--"), args)
    all_chunks = chunks(QMD)

    labels = if "--all" in args
        [l for (l, _) in all_chunks if l != "setup"]
    elseif isempty(wanted)
        error("usage: julia --project=. dev/figure.jl [--all] [--full] <label>...")
    else
        wanted
    end

    mkpath(OUTDIR)
    mkpath(PNGDIR)

    failed = String[]
    for label in labels
        println("\n=== $label")
        i = findfirst(p -> first(p) == label, all_chunks)
        if i === nothing
            println("  no such chunk label")
            push!(failed, label)
            continue
        end
        try
            inspect(label, all_chunks; full = full)
        catch err
            println("  FAILED: ", first(split(sprint(showerror, err), '\n')))
            push!(failed, label)
        end
    end

    isempty(failed) || println("\nfailed: ", join(failed, ", "))
    return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
