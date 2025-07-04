using Intercepts
using Random
using GLM
using Statistics
using CairoMakie

n = 100
p = 10000
k = 10

Random.seed!(1234)

μ0 = 0.95

X, y = generatedata(
  n, p; response=:binomial, μ0=μ0, x_type=:normal, x_density=0.9, ρ=0.3, s=k, amplitude=1, means=:random
)

reg = 0.01
randomize = false
freq = 1
maxit = 1000

res_grad = cdsolver(X, y, reg, lossfun=Logistic(), intercept_strategy=GradientStrategy(), maxit=maxit, randomize=randomize, update_freq=freq)
res_newt = cdsolver(X, y, reg, lossfun=Logistic(), intercept_strategy=NewtonStrategy(), maxit=maxit, randomize=randomize, update_freq=freq)
res_exact = cdsolver(X, y, reg, lossfun=Logistic(), intercept_strategy=ExactStrategy(), maxit=maxit, randomize=randomize, update_freq=freq)

fig = Figure()
ax = Axis(
  fig[1, 1],
  yscale=log
)

lines!(ax, res_grad.time, res_grad.gaps, label="Gradient")
lines!(ax, res_newt.time, res_newt.gaps .+ 1e-13, label="Newton")
lines!(ax, res_exact.time, res_exact.gaps, label="Exact")
fig[1, 2] = Legend(fig, ax)

fig

hcat(res_newt.primals[end], res_exact.primals[end], res_grad.primals[end])
hcat(res_newt.duals[end], res_exact.duals[end], res_grad.duals[end])
