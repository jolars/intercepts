module Intercepts

include("loss.jl")
include("intercept_strategies.jl")
include("cdsolver.jl")
include("gdsolver.jl")
include("math.jl")
include("data.jl")
include("normalize.jl")
include("multinomial_cdsolver.jl")
include("irlssolver.jl")
include("experiment.jl")

export cdsolver
export multinomial_cdsolver
export gdsolver
export irlssolver
export IRLSWeights
export LocalWeights
export MajorizedWeights
export NoIntercept
export InterceptStrategy
export GradientStrategy
export NewtonStrategy
export DampedNewtonStrategy
export BacktrackingGradientStrategy
export ExactStrategy
export update_intercept
export geomspace
export rescalecoefs
export hessian
export dual
export lambdamax
export gradient
export normalizefeatures
export LossFunction
export QuadraticLoss
export LogisticLoss
export PoissonLoss
export MultinomialLogisticLoss
export softmax_probs
export residual
export link
export invlink
export weight
export loss
export workingresponse
export generatedata
export simulated_experiment
export real_experiment

end
