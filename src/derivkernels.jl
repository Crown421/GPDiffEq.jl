# ToDo: nicer implementation
export DerivativeKernel

abstract type AbstractDerivativeKernel <: Kernel end

struct DerivativeKernel{Td01,Td10,Td11,Tk<:Kernel} <: AbstractDerivativeKernel
    d10::Td01
    d01::Td10
    d11::Td11

    function DerivativeKernel(kernel)
        d10(t1, t2) = first(first(Zygote.gradient(t1 -> kernel(t1, t2), t1)))
        d01(t1, t2) = first(first(Zygote.gradient(t2 -> kernel(t1, t2), t2)))
        d11(t1, t2) = first(Zygote.hessian(t -> kernel(t[1], t[2]), [t1, t2])[2])
        return new{typeof(d01),typeof(d10),typeof(d11),typeof(kernel)}(d10, d01, d11)
    end
end

#  perhaps not super clear, but this is probably the most often needed part
function (dk::DerivativeKernel)(t1, t2)
    return dk.d10(t1, t2)
end

# need tests, for various kernels and maybe a finitediff method to compare?

# function build_deriv_model(gp::AbstractGPs.PosteriorGP)
#     kernel = FirstCompDerivKernel(gp.prior.kernel)
#     f = GP(kernel)
#     # ToDo: This is brittle, needs to take into account mean
#     return AbstractGPs.PosteriorGP(f, gp.data)
# end