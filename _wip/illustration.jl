using Intercepts
using Random
using GLM
using CairoMakie

Random.seed!(134)
n = 100
p = 1
k = 10
μ0 = 0.9

X, y = generatedata(
  n, p; response=:binomial, μ0=μ0, x_type=:normal, x_density=0.9, ρ=0.99, s=k, amplitude=1,
)

strategies = [
  (strategy=GradientStrategy(), name="Gradient"),
  (strategy=NewtonStrategy(), name="Newton"),
  (strategy=ExactStrategy(), name="Exact")
]

reg = 0.5

results = [
  cdsolver(
    X,
    y,
    reg,
    lossfun=Logistic(),
    intercept_strategy=s.strategy,
    save_history=true,
  ) for s in strategies
]

λ = results[1].λ

β0 = range(-0.5, 3.5, length=100)
β = range(-0.5, 1.2, length=100)
grid = [(b0, b) for b0 in β0, b in β]

primal_values = zeros(size(grid))

for (i, (b0, b)) in enumerate(grid)
  η = b0 .+ X[:, 1] * b
  primal_values[i] = loss(Logistic(), η, y) + λ * abs(b)
end

fig = Figure()
ax = Axis(
  fig[1, 1],
  xlabel="Intercept (β₀)",
  ylabel="Coefficient (β₁)",
)

levels = geomspace(minimum(primal_values), maximum(primal_values), 20)

contour!(β0, β, primal_values, levels=levels, colormap=:grays)

for (res, s) in zip(results, strategies)
  scatterlines!(res.intercepts, dropdims(res.coefs, dims=1), label=s.name)
end

fig[1, 2] = Legend(fig, ax)

fig
