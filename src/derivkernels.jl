# ToDo: nicer implementation

abstract type AbstractDerivativeKernel <: Kernel end

struct FirstComponentDerivativeKernel{Tk<:Kernel} <: AbstractDerivativeKernel
    grad::Function

    function FirstCompDerivKernel(kernel)
        dker(t1, t2) = first(first(Zygote.gradient(t1 -> kernel(t1, t2), t1)))
        return new{typeof(kernel)}(dker)
    end
end

(dk::FirstCompDerivKernel)(x, y) = dk.grad(x, y)

struct SecondComponentDerivativeKernel{Tk<:Kernel} <: AbstractDerivativeKernel
    grad::Function

    function SecondCompDerivativeKernel(kernel)
        kerd(t1, t2) = first(first(Zygote.gradient(t2 -> kernel(t1, t2), t2)))
        return new{typeof(kernel)}(kerd)
    end
end

(dk::SecondCompDerivativeKernel)(x, y) = dk.grad(x, y)

struct BothComponentDerivativeKernel{Tk<:Kernel} <: AbstractDerivativeKernel
    grad::Function

    function BothCompDerivativeKernel(kernel)
        function dkerd(t1, t2)
            return first(Zygote.hessian(t -> ker(t[1], t[2]), [t1, t2])[2])
        end
        return new{typeof(kernel)}(dkerd)
    end
end

(dk::BothCompDerivativeKernel)(x, y) = dk.grad(x, y)

# need tests, for various kernels and maybe a finitediff method to compare?

# function build_deriv_model(gp::AbstractGPs.PosteriorGP)
#     kernel = FirstCompDerivKernel(gp.prior.kernel)
#     f = GP(kernel)
#     # ToDo: This is brittle, needs to take into account mean
#     return AbstractGPs.PosteriorGP(f, gp.data)
# end