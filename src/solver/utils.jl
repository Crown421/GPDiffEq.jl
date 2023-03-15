
# All of the following functions are fixes of the (in this context) slightly clunky MultiOutput interface of AbstractGPs.

## mean
function _mean(gp::AbstractGPs.PosteriorGP, x::AbstractVector{<:Real})
    xMO = _makeMOInput([x], gp.data.x)
    # println(xMO)
    return _fix_output(mean(gp, xMO))
end

# ToDo: make nicer
_fix_output(res::AbstractVector{<:Real}) = res
# ToDo: Need to figure out why dgp return `Vector{Any}`
_fix_output(res::AbstractVector{<:Any}) = permutedims(reduce(hcat, res))

# now the same for real scalars
function _mean(gp, x::Real)
    return only(mean(gp, [x]))
end

## cov
function _cov(gp, x::AbstractVector{<:Real})
    return cov(gp, x)
end

function _cov(gp, x::AbstractVector{<:AbstractVector{<:Real}})
    xMO = _makeMOInput(x, gp.data.x)
    return cov(gp, xMO)
end

function _cov(
    gp,
    x::AbstractVector{<:AbstractVector{<:Real}},
    y::AbstractVector{<:AbstractVector{<:Real}},
)
    xMO = _makeMOInput(x, gp.data.x)
    yMO = _makeMOInput(y, gp.data.x)
    return cov(gp, xMO, yMO)
end

# this should probably not be special case, but need to think about it more:
# function _mean(gp::AbstractGPs.PosteriorGP{<:DerivativeGP}, x::AbstractVector{<:Real})
#     xMO = _makeMOInput([x], gp.data.x)
#     dres = mean(gp, xMO)
#     return permutedims(reduce(hcat, dres))
# end

function _makeMOInput(x, xMO::KernelFunctions.MOInputIsotopicByFeatures)
    return KernelFunctions.MOInputIsotopicByFeatures(x, xMO.out_dim)
end
function _makeMOInput(x, xMO::KernelFunctions.MOInputIsotopicByOutputs)
    return KernelFunctions.MOInputIsotopicByOutputs(x, xMO.out_dim)
end

# speculative
function _makeMOInput(x, xMO)
    return only.(x)
end