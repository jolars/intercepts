using Statistics

"""
    normalizefeatures(x, normalization = :standardize; center = true)

Normalize feature columns and return `(x_out, centers, scales)`. With
`:standardize`, columns are centered and divided by their population standard
deviation; with `:none`, `x` is unchanged. Set `center = false` to scale without
centering, as required when fitting a model without an intercept. Constant
columns use a scale of one.

Sparse matrices retain their sparsity: only scaling is applied to `x_out`,
and callers must account for `centers ./ scales` implicitly.
"""
function normalizefeatures(
        x::AbstractMatrix,
        normalization::Symbol = :standardize;
        center::Bool = true,
    )
    p = size(x, 2)

    if normalization == :none
        return x, zeros(p)', ones(p)'
    elseif normalization == :standardize
        centers = mean(x; dims = 1)
        scales = stdm(x, centers; corrected = false, dims = 1)
        if !center
            centers .= 0
        end
    else
        throw(ArgumentError("Unsupported normalization method: $normalization"))
    end

    x_out = copy(x)

    scales[scales .== 0] .= 1.0 # Avoid division by zero

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

"""
    rescalecoefs(coefs, intercept, centers, scales; fit_intercept = true)

Undo feature normalization and return the intercept and coefficients on the
original feature scale. When `fit_intercept` is false, only the coefficients
are rescaled; feature normalization must omit centering in that case.
"""
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

    for j in 1:p
        coefs_rescaled[j] /= scales[j]
        x_bar_beta_sum += centers[j] * coefs_rescaled[j]
    end

    if fit_intercept
        intercept_rescaled -= x_bar_beta_sum
    end

    return intercept_rescaled, coefs_rescaled
end
