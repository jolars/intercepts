using SparseArrays
using JSON3

"""
    load_libsvm(path; n_samples, n_features, dense = false)

Parse a LIBSVM-format sparse data file into a feature matrix and a response
vector. Each line is `label idx:val idx:val ...`; feature indices are 1-based by
default but a 0-based file is detected and shifted automatically. `n_samples`
and `n_features` fix the returned dimensions so all-zero trailing rows and
never-observed feature columns are preserved exactly --- they are the canonical
dimensions recorded for each dataset in `data/libsvm/manifest.json`.

Return `(X, y)` with `X::SparseMatrixCSC{Float64,Int}` (dense `Matrix{Float64}`
when `dense = true`) and `y::Vector{Float64}` of the raw labels. The parse
mirrors `LIBSVMdata.load_dataset` exactly so cached experiment results stay
reproducible after the switch from run-time fetching to shipped inputs.
"""
function load_libsvm(
    path::AbstractString;
    n_samples::Integer,
    n_features::Integer,
    dense::Bool = false,
)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    y = Vector{Float64}(undef, n_samples)
    idx_start = 1

    open(path, "r") do io
        for (i, line) in enumerate(eachline(io))
            elements = split(line, " "; limit = 2)
            length(elements) == 1 && push!(elements, "")
            label, features = elements
            y[i] = parse(Float64, label)

            for feature in split(features, " ")
                isempty(feature) && continue
                idx_str, val_str = split(feature, ":")
                idx = parse(Int, idx_str)
                val = parse(Float64, val_str)
                idx == 0 && (idx_start = 0)
                if val != 0.0
                    push!(rows, i)
                    push!(cols, idx - idx_start + 1)
                    push!(vals, val)
                end
            end
        end
    end

    X = sparse(rows, cols, vals, n_samples, n_features)
    return dense ? Matrix(X) : X, y
end

"""
    intercepts_datadir()

Directory holding the shipped real-data inputs --- `data/` at the repository
root (the `Intercepts` package lives in `src/`), overridable with the
`INTERCEPTS_DATA_DIR` environment variable.
"""
intercepts_datadir() =
    get(ENV, "INTERCEPTS_DATA_DIR", normpath(joinpath(@__DIR__, "..", "data")))

"""
    load_local_dataset(name; dense = false)

Load a LIBSVM dataset from the shipped `data/libsvm/` tree rather than fetching
it at run time. The on-disk filename and the canonical `(n_samples, n_features)`
come from the committed `data/libsvm/manifest.json`; the raw input itself is
retrieved once with `experiments/fetch-data.sh` (see the README). A drop-in
replacement for `LIBSVMdata.load_dataset` returning `(X, y)`.
"""
function load_local_dataset(name::AbstractString; dense::Bool = false)
    dir = joinpath(intercepts_datadir(), "libsvm")
    manifest_path = joinpath(dir, "manifest.json")
    isfile(manifest_path) || error(
        "LIBSVM dataset manifest not found at $manifest_path. Retrieve the " *
        "real-data inputs first with `bash experiments/fetch-data.sh` (see the " *
        "README's \"Retrieve the real-data inputs\" section).",
    )

    manifest = JSON3.read(read(manifest_path, String))
    key = Symbol(name)
    haskey(manifest, key) || error(
        "Dataset \"$name\" is not listed in $manifest_path. Available: " *
        join(string.(keys(manifest)), ", "),
    )

    entry = manifest[key]
    path = joinpath(dir, String(entry.file))
    isfile(path) || error(
        "LIBSVM input \"$(entry.file)\" for dataset \"$name\" not found at $path. " *
        "Retrieve the real-data inputs first with `bash experiments/fetch-data.sh`.",
    )

    return load_libsvm(
        path;
        n_samples = Int(entry.n_samples),
        n_features = Int(entry.n_features),
        dense = dense,
    )
end
