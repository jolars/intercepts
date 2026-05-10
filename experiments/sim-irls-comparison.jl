using Intercepts
using Random
using DrWatson
using ProjectRoot
using JLD2

Random.seed!(2025)

# Imbalanced logistic ($\mu_0 = 0.99$): the gradient strategy fails (Lemma 3.3,
# fraction H_00/L_0 -> 0), and the IRLS subsection predicts that MM-IRLS realizes
# the same gradient-style update on the original loss while local-weight IRLS
# does not.
param_dict = Dict{String,Any}(
    "it" => collect(1:3),
    "n" => [500],
    "p" => [1000],
    "s" => [10],
    "reg" => [0.05],
    "μ0" => [0.99],
    "method" => [:direct_newton, :direct_gradient, :local_irls, :mm_irls],
)
params = dict_list(param_dict)

method_label = Dict(
    :direct_newton => "Direct-CD + Newton",
    :direct_gradient => "Direct-CD + Gradient",
    :local_irls => "Local-IRLS",
    :mm_irls => "MM-IRLS",
)

function run_method(method, X, y, reg)
    if method == :direct_newton
        return cdsolver(
            X,
            y,
            reg;
            lossfun = LogisticLoss(),
            intercept_strategy = NewtonStrategy(),
            maxit = 500,
            randomize = true,
        )
    elseif method == :direct_gradient
        return cdsolver(
            X,
            y,
            reg;
            lossfun = LogisticLoss(),
            intercept_strategy = GradientStrategy(),
            maxit = 500,
            randomize = true,
        )
    elseif method == :local_irls
        return irlssolver(
            X,
            y,
            reg;
            lossfun = LogisticLoss(),
            weights = LocalWeights(),
            intercept_strategy = NewtonStrategy(),
            max_outer = 200,
            max_inner = 30,
            randomize = true,
        )
    elseif method == :mm_irls
        return irlssolver(
            X,
            y,
            reg;
            lossfun = LogisticLoss(),
            weights = MajorizedWeights(),
            intercept_strategy = NewtonStrategy(),
            max_outer = 200,
            max_inner = 30,
            randomize = true,
        )
    else
        error("Unknown method $method")
    end
end

results = []

for d in params
    @unpack it, n, p, s, reg, μ0, method = d

    Random.seed!(it)
    X, y = generatedata(
        n,
        p;
        response = :binomial,
        μ0 = μ0,
        x_type = :normal,
        ρ = 0.6,
        s = s,
        amplitude = 1.0,
    )

    res = run_method(method, X, y, reg)

    d_exp = Dict{String,Any}(copy(d))
    d_exp["strategy"] = method_label[method]
    d_exp["time"] = res.time
    d_exp["gaps"] = res.gaps
    d_exp["relgaps"] = res.relgaps
    d_exp["primals"] = res.primals
    push!(results, d_exp)
end

outfile = @projectroot("results", "sim-irls-comparison.jld2")
@save outfile results
