
function _mean(gp::AbstractGPs.AbstractGP, x::AbstractVector{<:Real})
    return mean(gp, MOInput([x], length(x)))
end

# now the same for real scalars
function _mean(gp, x::Real)
    return only(mean(gp, [x]))
end