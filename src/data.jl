using Random
using LinearAlgebra
using SparseArrays
using GLM

function generatedata(
  n::Int,
  p::Int
  ;
  response::Symbol=:normal,
  μ0::Real=0,
  x_density::Real=0.1,
  x_type=:normal,
  ρ::Real=0,
  s::Int=5,
  type::Symbol=:constant,
  amplitude::Real=1,
)
  s = min(s, p)
  β = zeros(p)

  ind = Int.(round.(range(1, p; length=s)))

  if type == :constant
    β[ind] .= 1
  elseif type == :rnorm
    β[ind] .= randn(s)
  elseif type == :uniform
    β[ind] .= rand(s) * 2 .- 1
  end

  β .*= amplitude

  if x_type == :normal
    x = randn(n, p)

    if ρ != 0
      inds = 1:p
      Σ = Symmetric(ρ .^ abs.(inds .- inds'))
      chol = cholesky(Σ)
      x = x * chol.L
    end
  elseif x_type == :binary
    x = Int.(sprand(Bool, n, p, Float64(x_density)))

    if ρ != 0
      for j in 2:p
        for i in 1:n
          if rand() < ρ
            x[i, j] = x[i, j-1]
          end
        end
      end
    end
  else
    throw(ArgumentError("Unsupported x_type: $x_type"))
  end

  η = x * β

  # Create distribution object
  if response == :normal
    dist = Normal()
    link = canonicallink(dist)
  elseif response == :binomial
    dist = Bernoulli()
    link = canonicallink(dist)
  elseif response == :poisson
    dist = Poisson()
    link = canonicallink(dist)
  else
    throw(ArgumentError("Unsupported family: $response"))
  end

  β0 = GLM.linkfun.(link, μ0)
  η .+= β0
  μ = GLM.linkinv.(link, η)

  y = Vector{eltype(μ)}(undef, n)

  for i in 1:n
    ddist = if response == :normal
      Normal(μ[i], 1)
    elseif response == :binomial
      Bernoulli(μ[i])
    elseif response == :poisson
      Poisson(μ[i])
    end
    y[i] = rand(ddist)
  end

  return x, y
end
