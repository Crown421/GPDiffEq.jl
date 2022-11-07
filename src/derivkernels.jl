# ToDo: nicer implementation

abstract type AbstractDerivativeKernel <: Kernel end

struct DerivativeKernel{Tf,Tk<:Kernel} <: AbstractDerivativeKernel
    d01::Tf
    d10::Tf
    d11::Tf

    function FirstCompDerivKernel(kernel)
        d01(t1, t2) = first(first(Zygote.gradient(t1 -> kernel(t1, t2), t1)))
        d10(t1, t2) = first(first(Zygote.gradient(t2 -> kernel(t1, t2), t2)))
        d11(t1, t2) = first(Zygote.hessian(t -> ker(t[1], t[2]), [t1, t2])[2])
        return new{typeof(d01),typeof(kernel)}(d01, d10, d11)
    end
end

# need tests, for various kernels and maybe a finitediff method to compare?

# function build_deriv_model(gp::AbstractGPs.PosteriorGP)
#     kernel = FirstCompDerivKernel(gp.prior.kernel)
#     f = GP(kernel)
#     # ToDo: This is brittle, needs to take into account mean
#     return AbstractGPs.PosteriorGP(f, gp.data)
# end