## Author-only: assemble the exact real-data inputs into one upload-ready
## archive (`intercepts-data-<version>.tar.gz`) for deposit on Zenodo, and write
## the committed manifest/checksum files that reproducers verify against.
##
## This is the ONLY step that fetches data from upstream mirrors. Run it once,
## upload the resulting tarball to Zenodo, then fill the minted DOI into
## `experiments/fetch-data.sh` and the README. Reproducers never run this; they
## run `experiments/fetch-data.sh`, which downloads the pinned archive instead.
##
##   julia --project=. experiments/build-data-archive.jl
##
## Inputs gathered:
##   - LIBSVM datasets (w1a, news20.binary, shuttle.scale[.t], a4a, leukemia,
##     breast-cancer, gisette_scale), fetched + decompressed via LIBSVMdata and
##     copied verbatim into data/libsvm/.
##   - Yeoh2002.rds (pediatric ALL gene-expression panel) under data/yeoh/.
##   - congress109 phrase counts, snapshotted from R's textir via
##     experiments/snapshot-congress109.R into data/congress109/.

using LIBSVMdata
using ProjectRoot
using JSON3

const ARCHIVE_VERSION = "v1"

# Dataset name (as passed to load_local_dataset in the experiments) => filename
# shipped under data/libsvm/. Names match the keys used at every call site.
const LIBSVM_DATASETS = [
    "w1a" => "w1a",
    "news20.binary" => "news20.binary",
    "shuttle.scale" => "shuttle.scale",
    "shuttle.scale.t" => "shuttle.scale.t",
    "a4a" => "a4a",
    "leukemia" => "leukemia",
    "breast-cancer" => "breast-cancer",
    "gisette_scale" => "gisette_scale",
]

sha256_of(path) = first(split(read(`sha256sum $path`, String)))

# LIBSVMdata stores the raw download as DATASETS[name][:file] and, when that is
# compressed, the decompressed sibling alongside it. Return the decompressed
# path that load_dataset actually parses.
function decompressed_source(home, file)
    if endswith(file, ".bz2")
        return joinpath(home, file[1:(end - 4)])
    elseif endswith(file, ".tar.xz")
        return joinpath(home, file[1:(end - 7)])
    elseif endswith(file, ".xz")
        return joinpath(home, file[1:(end - 3)])
    else
        return joinpath(home, file)
    end
end

datadir = @projectroot("data")
libsvm_out = joinpath(datadir, "libsvm")
mkpath(libsvm_out)

home = LIBSVMdata.get_dataset_home()
catalogue = LIBSVMdata.get_datasets()

manifest = Dict{String, Any}()

for (name, shipped) in LIBSVM_DATASETS
    haskey(catalogue, name) || error("Unknown LIBSVMdata dataset: $name")
    meta = catalogue[name]
    m, n = meta[:dims]

    src = decompressed_source(home, meta[:file])
    if !isfile(src)
        # Trigger download + decompression into the LIBSVMdata home.
        println("Fetching $name via LIBSVMdata ...")
        LIBSVMdata.load_dataset(name; verbose = false)
    end
    isfile(src) || error("Expected decompressed LIBSVM file not found: $src")

    dst = joinpath(libsvm_out, shipped)
    cp(src, dst; force = true)
    println("  copied $(basename(src)) -> data/libsvm/$shipped")

    manifest[name] = Dict(
        "file" => shipped,
        "n_samples" => m,
        "n_features" => n,
        "type" => meta[:type],
        "ncls" => meta[:ncls],
        "source" => join([LIBSVMdata.BASE_URL, meta[:type], meta[:file]], "/"),
    )
end

open(joinpath(libsvm_out, "manifest.json"), "w") do io
    JSON3.pretty(io, manifest)
    println(io)
end
println("Wrote data/libsvm/manifest.json")

# Yeoh2002: the raw RDS is the input of record; it must already be staged under
# data/yeoh/ (the repo ships it via the archive). fetch-yeoh.R converts it to
# CSV at retrieval time.
yeoh_rds = joinpath(datadir, "yeoh", "Yeoh2002.rds")
isfile(yeoh_rds) ||
    error(
        "data/yeoh/Yeoh2002.rds is missing. Stage it once from the IowaBiostat " *
            "collection (Yeoh2002.rds) before building the archive.",
    )
println("Using data/yeoh/Yeoh2002.rds")

# congress109: snapshot the textir phrase-count matrix into data/congress109/.
println("Snapshotting congress109 via R/textir ...")
run(`Rscript $(@projectroot("experiments", "snapshot-congress109.R"))`)

# Files shipped in the archive (paths relative to the repository root).
archived = String[
    "data/yeoh/Yeoh2002.rds",
    "data/congress109/counts.mtx",
    "data/congress109/phrases.txt",
    "data/libsvm/manifest.json",
]
for (_, shipped) in LIBSVM_DATASETS
    push!(archived, "data/libsvm/$shipped")
end

# Committed checksum manifest (sha256sum -c format, repo-root-relative paths).
repo = dirname(datadir)
open(joinpath(datadir, "MANIFEST.sha256"), "w") do io
    for rel in sort(archived)
        h = sha256_of(joinpath(repo, rel))
        println(io, "$h  $rel")
    end
end
println("Wrote data/MANIFEST.sha256")

# Build the upload-ready tarball at the repository root (gitignored).
tarball = "intercepts-data-$(ARCHIVE_VERSION).tar.gz"
members = vcat(archived, ["data/MANIFEST.sha256", "data/README.md"])
existing = filter(m -> isfile(joinpath(repo, m)), members)
run(Cmd(`tar -czf $tarball $existing`; dir = repo))
println(
    "Wrote $tarball (",
    join(["$(round(filesize(joinpath(repo, tarball)) / 1.0e6; digits = 1)) MB"]),
    ")",
)
println()
println("Next: upload $tarball to Zenodo, then put the DOI in")
println("  experiments/fetch-data.sh and README.md.")
