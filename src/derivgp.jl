"""
    DerivativeGP

The Gaussian Process (GP) 
"""
struct DerivativeGP{Tm<:MeanFunction,Tdk<:AbstractDerivativeKernel} <: AbstractGP
    dmean::Tm
    dkernel::Tdk

    function DerivativeGP(mean, kernel <: Kernel)
        dmean = _deriv_meanfunction(mean)
        dker = DerivativeKernel(kernel)
        return new{Tm,Tdk}(dmean, dker)
    end
end

# not sure about AbstractGPs annotations
function _deriv_meanfunction(
    ::Union{AbstractGPs.ZeroMean{T},AbstractGPs.ConstMean{T}}
) where {T}
    return AbstractGPs.ZeroMean{T}()
end
