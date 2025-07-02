using Test
using Intercepts
using Random
using GLM
using Statistics
using CairoMakie
using LinearAlgebra

n = 100
p = 1000
k = 10

Random.seed!(1234)

ρ = 0.5  # autocorrelation parameter
Σ = [ρ^abs(i - j) for i in 1:p, j in 1:p];
X = randn(n, p) * cholesky(Σ).L';

β = zeros(p);
β[1:k] = randn(k) * 2;

p0 = 0.1
β0 = log(p0 / (1 - p0));
η = X * β .+ β0;
pr = 1 ./ (1 .+ exp.(-η));
y = rand(n) .< pr;

# intercept_strategy = NoIntercept()

# res_lm = lasso(X, y, 0.0, loss_fun=Quadratic(), intercept_strategy=intercept_strategy, maxit=1000)
#
# # ols_fit = X \ y
# ols_fit = [ones(size(X, 1)) X] \ y
#
# y_bin = y .> mean(y)
#
# res_logreg = lasso(X, y_bin, 0.5, loss_fun=Logistic())
# ols_logreg = glm([ones(n) X], y_bin, Binomial(), LogitLink())

reg = 0.01

randomize = false
freq = 1

res_grad = cdsolver(X, y, reg, loss_fun=Logistic(), intercept_strategy=GradientStrategy(), maxit=5000, randomize=randomize, update_freq=freq)
res_newt = cdsolver(X, y, reg, loss_fun=Logistic(), intercept_strategy=NewtonStrategy(), maxit=5000, randomize=randomize, update_freq=freq)
res_exact = cdsolver(X, y, reg, loss_fun=Logistic(), intercept_strategy=ExactStrategy(), maxit=5000, randomize=randomize, update_freq=freq)

CairoMakie.activate!(type="svg")

# opt = minimum(hcat(res_grad.primals, res_newt.primals, res_exact.primals))

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

hcat(res_newt.primals[end], res_exact.primals[end], res_grad.primals[end])
hcat(res_newt.duals[end], res_exact.duals[end], res_grad.duals[end])
