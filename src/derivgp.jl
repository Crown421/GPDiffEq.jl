export DerivativeGP

"""
    DerivativeGP

The Gaussian Process (GP) 
"""
struct DerivativeGP{Tm<:AbstractGPs.MeanFunction,Tdk<:AbstractDerivativeKernel} <:
       AbstractGPs.AbstractGP
    dmean::Tm
    dkernel::Tdk

    function DerivativeGP(mean, kernel::Kernel)
        dmean = _deriv_meanfunction(mean)
        dker = DerivativeKernel(kernel)
        return new{typeof(dmean),typeof(dker)}(dmean, dker)
    end
end

# ToDo: print function for type

function DerivativeGP(kernel::Kernel)
    return DerivativeGP(AbstractGPs.ZeroMean(), kernel)
end

# not sure about AbstractGPs annotations
function _deriv_meanfunction(
    ::Union{AbstractGPs.ZeroMean{T},AbstractGPs.ConstMean{T}}
) where {T}
    return AbstractGPs.ZeroMean{T}()
end

# ToDo: Tests!
function _deriv_meanfunction(mean::AbstractGPs.CustomMean) where {T}
    # check for >1D
    df(x) = Zygote.gradient(x -> mean(x), x)
    return AbstractGPs.CustomMean(df)
end
