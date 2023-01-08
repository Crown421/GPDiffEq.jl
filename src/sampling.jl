# goal: A function that takes an AbstractGP and a vector of sample points (could be optional)
# - samples from the GP at those points
# - creates interpolation function
# - Challenge could be combining the 2D+ interpolations

# one function for the sample points (also exported, will start with that)
# xsample must be a regular grid for Interpolations.jl, need to do something about that. 

export sample_points, sample_function

function sample_points(gp::AbstractGPs.AbstractGP, xsample::AbstractVector)
    m = mean(gp, xsample)
    # gp.(xsample)
    ζ = randn(length(xsample))
    # K = Ks(xsample)
    # K = 0.5 * (K+ K') + 1e-12*I
    K = cov(gp, xsample) + 1e-12 * I

    # taken from https://github.com/JuliaGaussianProcesses/KernelFunctions.jl/blob/master/src/matrix/kernelpdmat.jl
    # could likely be better/ done as part of `mean_and_cov` if PDMat is loaded or so
    Kmax = maximum(K)
    α = eps(eltype(K))
    while !isposdef(K + α * I) && α < 0.01 * Kmax
        α *= 2.0
    end
    if α >= 0.01 * Kmax
        error(
            "Adding noise on the diagonal was not sufficient to build a positive-definite" *
            " matrix:\n\t- Check that your kernel parameters are not extreme\n\t- Check" *
            " that your data is sufficiently sparse\n\t- Maybe use a different kernel",
        )
    end
    Kchol = cholesky(K + α * I)

    return m .+ Kchol.L * ζ
end

# for MOInput, probably dispatch on AbstractVector{<:Tuple}?
function sample_function(
    gp::AbstractGPs.AbstractGP,
    xsample::AbstractVector;
    interpolationalg::Interpolations.InterpolationType=BSpline(Cubic(Line(OnGrid()))),
    extrapolation=Line(),
)
    sample = sample_points(gp, xsample)
    itp = interpolate(sample, interpolationalg)
    sitp = Interpolations.scale(itp, xsample)
    return extrapolate(sitp, extrapolation)
end