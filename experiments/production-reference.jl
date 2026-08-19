using Intercepts
using CSV
using DataFrames
using JSON3
using ProjectRoot

# Produce one independently certified objective reference for every problem in
# the production-solver figures. The production drivers report
# mean(loss) + lambda * ||beta||_1, whereas Intercepts uses the equivalent
# sum(loss) + n * lambda * ||beta||_1 scale internally. Dividing both primal
# and dual values by n puts the certificate on the plotted scale.

function reference_solve(problem, family, problem_dir)
    X = Matrix(DataFrame(CSV.File(joinpath(problem_dir, "X.csv"); header = false)))
    y_table = DataFrame(CSV.File(joinpath(problem_dir, "y.csv")))
    y = family == "logistic" ? Float64.(y_table.y01) : Float64.(y_table.y)
    meta = JSON3.read(read(joinpath(problem_dir, "meta.json"), String))
    reg = Float64(meta.reg)

    lossfun = family == "logistic" ? LogisticLoss() : PoissonLoss()
    result = cdsolver(
        X,
        y,
        reg;
        lossfun = lossfun,
        intercept_strategy = NewtonStrategy(),
        tol = 1.0e-10,
        maxit = 10_000,
        randomize = false,
        normalization = :none,
    )

    n = length(y)
    primal = result.primals[end] / n
    dual = result.duals[end] / n
    gap = primal - dual
    relative_gap = gap / max(abs(primal), 1.0e-15)
    relative_gap <= 1.0e-8 || error(
        "reference solve for $problem has relative duality gap $relative_gap",
    )

    return (
        problem = problem,
        family = family,
        lambda = Float64(meta.lambda),
        F_star = primal,
        dual_bound = dual,
        duality_gap = gap,
        relative_duality_gap = relative_gap,
        method = "Intercepts guarded-Newton CD",
    )
end

root = @projectroot()
logistic_root = joinpath(root, "results", "real-solvers")
poisson_root = joinpath(root, "results", "real-solvers-poisson", "congress109")

references = [
    reference_solve("synthetic", "logistic", logistic_root),
    reference_solve("w1a", "logistic", joinpath(logistic_root, "w1a")),
    reference_solve(
        "news20-3pct",
        "logistic",
        joinpath(logistic_root, "news20-3pct"),
    ),
    reference_solve("congress109", "poisson", poisson_root),
]

output = joinpath(root, "results", "production-references.csv")
CSV.write(output, DataFrame(references))
println("Wrote $output")
