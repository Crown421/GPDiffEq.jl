
function _mean(gp::AbstractGPs.PosteriorGP, x::AbstractVector{<:Real})
    # ToDo: Need to make this depend on the input type of the GP
    xMO = _makeMOInput([x], gp.data.x)
    return mean(gp, xMO)
end

# now the same for real scalars
function _mean(gp, x::Real)
    return only(mean(gp, [x]))
end

function _makeMOInput(x, xMO::KernelFunctions.MOInputIsotopicByFeatures)
    return KernelFunctions.MOInputIsotopicByFeatures(x, xMO.out_dim)
end
function _makeMOInput(x, xMO::KernelFunctions.MOInputIsotopicByOutputs)
    return KernelFunctions.MOInputIsotopicByOutputs(x, xMO.out_dim)
end