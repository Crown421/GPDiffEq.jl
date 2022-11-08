export DerivativeGP

"""
    DerivativeGP

The Gaussian Process (GP) 
"""
struct DerivativeGP{Tf,Tm<:AbstractGPs.MeanFunction,Tdkc<:DerivativeKernelCollection} <:
       AbstractGPs.AbstractGP
    f::Tf
    dmean::Tm
    dkernel::Tdkc

    function DerivativeGP(mean, kernel::Kernel)
        f = GP(mean, kernel)
        dmean = _deriv_meanfunction(mean)
        dkerc = DerivativeKernelCollection(kernel)
        return new{typeof(f),typeof(dmean),typeof(dkerc)}(f, dmean, dkerc)
    end
end

function DerivativeGP(kernel::Kernel)
    return DerivativeGP(AbstractGPs.ZeroMean(), kernel)
end

## printing
# ToDo: printing tests
function Base.show(io::IO, ::MIME"text/plain", m::DerivativeGP)
    return print(
        io, "Derivative of GP f: \n", "  mean: ", m.f.mean, "\n", "  kernel: ", m.f.kernel
    )
end
# not yet sure what I want here
# Base.show(io::IO, m::DerivativeGP) = print(io, m.x, '(', m.y, ')')

### mean function

# not sure about AbstractGPs annotations
function _deriv_meanfunction(
    ::Union{AbstractGPs.ZeroMean{T},AbstractGPs.ConstMean{T},T}
) where {T<:Real}
    return AbstractGPs.ZeroMean{T}()
end

function _deriv_meanfunction(mean::AbstractGPs.CustomMean) where {T}
    # check for >1D
    df(x) = Zygote.gradient(x -> mean(x), x)
    return AbstractGPs.CustomMean(df)
end

### AbstractGP interface implementation.

function Statistics.mean(f::DerivativeGP, x::AbstractVector)
    return AbstractGPs._map_meanfunction(f.dmean, x)
end

Statistics.cov(f::DerivativeGP, x::AbstractVector) = kernelmatrix(f.dkernel.d11, x)

Statistics.var(f::DerivativeGP, x::AbstractVector) = kernelmatrix_diag(f.dkernel.d11, x)

function Statistics.cov(f::DerivativeGP, x::AbstractVector, x′::AbstractVector)
    return kernelmatrix(f.dkernel.d11, x, x′)
end

### FiniteGPs
# no special implementation needed

function Base.show(io::IO, ::MIME"text/plain", m::AbstractGPs.FiniteGP{<:DerivativeGP})
    return print(
        io,
        "FiniteGP of Derivative of GP f: \n",
        "    mean: ",
        m.f.f.mean,
        "\n    kernel: ",
        m.f.f.kernel,
        "\n  x: ",
        m.x,
        "\n  Σy: ",
        m.Σy,
    )
end
