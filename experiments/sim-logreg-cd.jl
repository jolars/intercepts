using Intercepts
using Random
using GLM
using Statistics
using CairoMakie

n = 100
p = 1000
k = 10

Random.seed!(1234)

μ0 = 0.9

X, y = generatedata(
    n,
    p;
    response = :binomial,
    μ0 = μ0,
    x_type = :normal,
    x_density = 1,
    ρ = 0.6,
    s = k,
    amplitude = 1,
    means = :random,
)

reg = 0.02
randomize = false
freq = 1
maxit = 1000

res_grad = cdsolver(
    X,
    y,
    reg,
    lossfun = Logistic(),
    intercept_strategy = GradientStrategy(),
    maxit = maxit,
    randomize = randomize,
    update_freq = freq,
)
res_newt = cdsolver(
    X,
    y,
    reg,
    lossfun = Logistic(),
    intercept_strategy = NewtonStrategy(),
    maxit = maxit,
    randomize = randomize,
    update_freq = freq,
)
res_exact = cdsolver(
    X,
    y,
    reg,
    lossfun = Logistic(),
    intercept_strategy = ExactStrategy(),
    maxit = maxit,
    randomize = randomize,
    update_freq = freq,
)

fig = Figure()
ax = Axis(fig[1, 1], yscale = log)

lines!(ax, res_grad.time, res_grad.gaps, label = "Gradient")
lines!(ax, res_newt.time, res_newt.gaps, label = "Newton")
lines!(ax, res_exact.time, res_exact.gaps, label = "Exact")
fig[1, 2] = Legend(fig, ax)

fig

hcat(res_newt.primals[end], res_exact.primals[end], res_grad.primals[end])
hcat(res_newt.duals[end], res_exact.duals[end], res_grad.duals[end])
hcat(res_newt.gaps[end], res_exact.gaps[end], res_grad.gaps[end])
