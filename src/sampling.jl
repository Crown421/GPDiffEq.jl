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
    K = cov(gp, xsample)

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

    ys = m .+ Kchol.L * ζ
    return reshape_isotopic_multi_output(ys, gp)
end

# to cover SISO case for `reshape_isotopic_multi_output`
_reshape_imo(x, y) = y

# the challenge lies in the may possible options: 
# need to write them down, not just code into the void. 
# Want to feed ranges
# Then generate points (which needs a vector for 1D, and MOInputs for >1D)
# Some splatting, via collect(Iterators.product). 
# Really only two cases, but should write down what each step in each case needs, and more critically, write a test script!

# ToDo: TESTS!
# not sure about returning the collected points, but tbd
function sample_points(gp::AbstractGPs.AbstractGP, xsample::AbstractRange)
    x = collect(xsample)
    return sample_points(gp, x)
end

function sample_points(gp::AbstractGPs.AbstractGP, xsample::AbstractVector{<:AbstractRange})
    x = collect.(Iterators.product(xsample...))[:]
    x_mo = _makeMOInput(x, gp.data.x)
    return sample_points(gp, x_mo)
end

function _makeMOInput(x, xMO::KernelFunctions.MOInputIsotopicByFeatures)
    return KernelFunctions.MOInputIsotopicByFeatures(x, xMO.out_dim)
end
function _makeMOInput(x, xMO::KernelFunctions.MOInputIsotopicByOutputs)
    return KernelFunctions.MOInputIsotopicByOutputs(x, xMO.out_dim)
end

# for MOInput, probably dispatch on AbstractVector{<:Tuple}?
function sample_function(
    gp::AbstractGPs.AbstractGP,
    xsample;
    interpolationalg::Interpolations.InterpolationType=BSpline(Quadratic(Line(OnGrid()))),
    extrapolation=Line(),
)
    # xsample is a range, need to generate MOInputs from it
    sample = sample_points(gp, xsample)
    esitp = _create_itps(xsample, sample, interpolationalg, extrapolation)
    return esitp
end

# Implementing for ranges below, which slightly breaks my previous scripts, but is what I did in practice, and is more efficient. 
# Maybe later implement for general xsample inputs on (potentially) irregular grids. 
# https://juliamath.github.io/Interpolations.jl/latest/control/#Gridded-interpolation

# This might not be the best dispatch, but will do for now.
function _create_itps(xsample, sample, interpolationalg, extrapolation)
    itp = interpolate(sample, interpolationalg)
    sitp = _scale_itp(itp, xsample)
    esitp = extrapolate(sitp, extrapolation)
    return esitp
end

function _scale_itp(itp, xsample::AbstractVector{<:AbstractRange})
    return Interpolations.scale(itp, xsample...)
end
_scale_itp(itp, xsample::AbstractRange) = Interpolations.scale(itp, xsample)

# Now for Multi-Output systems, create an multi-output interpolation function
function _create_itps(
    xsample::AbstractVector{<:AbstractRange},
    sample::AbstractVector{<:AbstractVector},
    interpolationalg,
    extrapolation,
)
    itps = [
        _create_itps(xsample, _reshape(xsample, y), interpolationalg, extrapolation) for
        y in eachrow(sample.X)
    ]
    f(x) = [_itp(x...) for _itp in itps]
    return f
end

function _reshape(xsample, y)
    # might need a collect
    return reshape(y, length.(xsample)...)
end
