module Intercepts

include("loss.jl")
include("intercept_strategies.jl")
include("cdsolver.jl")
include("gdsolver.jl")
include("math.jl")
include("data.jl")

export cdsolver
export gdsolver
export NoIntercept
export InterceptStrategy
export GradientStrategy
export NewtonStrategy
export ExactStrategy
export update_intercept
export hessian
export dual
export gradient
export LossFunction
export Quadratic
export Logistic
export residual
export link
export invlink
export weight
export loss
export workingresponse
export generatedata

end
