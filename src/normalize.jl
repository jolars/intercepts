using Statistics

function normalizefeatures(x::AbstractMatrix, normalization::Symbol=:standardize)
  p = size(x, 2)

  if normalization == :none
    return zeros(p)', ones(p)'
  elseif normalization == :standardize
    centers = mean(x, dims=1)
    scales = stdm(x, centers, corrected=false, dims=1)
  else
    throw(ArgumentError("Unsupported normalization method: $normalization"))
  end

  x_out = copy(x)

  if issparse(x)
    for j in 1:p
      x_out[:, j] ./= scales[j]
    end
  else
    x_out .-= centers
    x_out ./= scales
  end

  return x_out, centers, scales
end
