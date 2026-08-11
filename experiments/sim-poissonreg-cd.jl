using Intercepts
using Random
using GLM
using CairoMakie

n = 100 # enough samples for stable fits
p = 1000 # moderate dimensionality
k = 20 # nonzeros in β
μ0 = 3.0 # baseline mean count on response scale

X, y = generatedata(n, p;
    response = :poisson,  # Poisson GLM
    μ0 = μ0,        # log link will use log(μ0)
    x_type = :normal,
    ρ = 0.2,       # mild feature correlation
    s = k,         # sparsity
    type = :rnorm,    # random nonzero β
    amplitude = 0.12,      # keeps η small so μ ≈ 1–10
)

reg = 0.5
maxit = 10000

res_newt = cdsolver(
    X,
    y,
    reg,
    lossfun = PoissonLoss(),
    intercept_strategy = NewtonStrategy(),
    maxit = maxit,
)
res_exact = cdsolver(
    X,
    y,
    reg,
    lossfun = PoissonLoss(),
    intercept_strategy = ExactStrategy(),
    maxit = maxit,
)

fig = Figure()
ax = Axis(fig[1, 1], yscale = log)

lines!(ax, res_newt.time, res_newt.gaps, label = "Newton")
lines!(ax, res_exact.time, res_exact.gaps, label = "Exact")
fig[1, 2] = Legend(fig, ax)

fig
