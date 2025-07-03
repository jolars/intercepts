using Intercepts
using Random
using GLM
using CairoMakie

n = 100
p = 1000
k = 10

Random.seed!(1234)

μ0 = 0.9

X, y = generatedata(
  n, p; response=:binomial, μ0=μ0, x_type=:binary, x_density=1, ρ=0.9, s=k, amplitude=0.1
)

reg = 0.05
maxit = 10000

res_grad = gdsolver(X, y, reg, lossfun=Logistic(), intercept_strategy=GradientStrategy(), maxit=maxit)
res_newt = gdsolver(X, y, reg, lossfun=Logistic(), intercept_strategy=NewtonStrategy(), maxit=maxit)
res_exact = gdsolver(X, y, reg, lossfun=Logistic(), intercept_strategy=ExactStrategy(), maxit=maxit)

fig = Figure()
ax = Axis(
  fig[1, 1],
  yscale=log
)

lines!(ax, res_grad.time, res_grad.gaps, label="Gradient")
lines!(ax, res_newt.time, res_newt.gaps, label="Newton")
lines!(ax, res_exact.time, res_exact.gaps, label="Exact")
fig[1, 2] = Legend(fig, ax)

fig
