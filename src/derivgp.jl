export DerivativeGP

"""
    DerivativeGP

The Gaussian Process (GP) 
"""
struct DerivativeGP{Tm<:AbstractGPs.MeanFunction,Tdkc<:DerivativeKernelCollection} <:
       AbstractGPs.AbstractGP
    dmean::Tm
    dkernel::Tdkc

    function DerivativeGP(mean, kernel::Kernel)
        dmean = _deriv_meanfunction(mean)
        dkerc = DerivativeKernelCollection(kernel)
        return new{typeof(dmean),typeof(dkerc)}(dmean, dkerc)
    end
end

# ToDo: print function for type

function DerivativeGP(kernel::Kernel)
    return DerivativeGP(AbstractGPs.ZeroMean(), kernel)
end

# not sure about AbstractGPs annotations
function _deriv_meanfunction(
    ::Union{AbstractGPs.ZeroMean{T},AbstractGPs.ConstMean{T},T}
) where {T<:Real}
    return AbstractGPs.ZeroMean{T}()
end

# ToDo: Tests!
function _deriv_meanfunction(mean::AbstractGPs.CustomMean) where {T}
    # check for >1D
    df(x) = Zygote.gradient(x -> mean(x), x)
    return AbstractGPs.CustomMean(df)
end

# AbstractGP interface implementation.

function Statistics.mean(f::DerivativeGP, x::AbstractVector)
    return AbstractGPs._map_meanfunction(f.dmean, x)
end

Statistics.cov(f::DerivativeGP, x::AbstractVector) = kernelmatrix(f.dkernel.d11, x)

Statistics.var(f::DerivativeGP, x::AbstractVector) = kernelmatrix_diag(f.dkernel.d11, x)

function Statistics.cov(f::DerivativeGP, x::AbstractVector, x′::AbstractVector)
    return kernelmatrix(f.dkernel.d11, x, x′)
end
