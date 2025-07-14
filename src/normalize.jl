using Statistics

function normalizefeatures(x::AbstractMatrix, normalization::Symbol = :standardize)
    p = size(x, 2)

    if normalization == :none
        return zeros(p)', ones(p)'
    elseif normalization == :standardize
        centers = mean(x; dims = 1)
        scales = stdm(x, centers; corrected = false, dims = 1)
    else
        throw(ArgumentError("Unsupported normalization method: $normalization"))
    end

    x_out = copy(x)

    scales[scales.==0] .= 1.0  # Avoid division by zero

    if issparse(x)
        for j = 1:p
            x_out[:, j] ./= scales[j]
        end
    else
        x_out .-= centers
        x_out ./= scales
    end

    return x_out, centers, scales
end

function rescalecoefs(
    coefs::AbstractVector,
    intercept::Real,
    centers::AbstractMatrix,
    scales::AbstractMatrix;
    fit_intercept::Bool = true,
)
    p = length(coefs)
    coefs_rescaled = copy(coefs)
    intercept_rescaled = intercept

    x_bar_beta_sum = 0

    for j = 1:p
        coefs_rescaled[j] /= scales[j]
        x_bar_beta_sum += centers[j] * coefs_rescaled[j]
    end

    if fit_intercept
        intercept_rescaled -= x_bar_beta_sum
    end

    return intercept_rescaled, coefs_rescaled
end
