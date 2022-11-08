# ToDo: nicer implementation
export DerivativeKernelCollection

abstract type AbstractDerivativeKernel <: Kernel end

# ToDo: finite differencing for Matern kernels, and in particular transformed matern kernels
# first(FiniteDiff.finite_difference_gradient(t1 -> ker(t1[1], t2), [t1]))
struct FirstComponentDerivativeKernel{Tdk,Tk<:Kernel} <: AbstractDerivativeKernel
    dk10::Tdk

    function FirstComponentDerivativeKernel(kernel)
        d10(t1, t2) = first(first(Zygote.gradient(t1 -> kernel(t1, t2), t1)))
        return new{typeof(d10),typeof(kernel)}(d10)
    end
end

(dk::FirstComponentDerivativeKernel)(t1, t2) = dk.dk10(t1, t2)

struct SecondComponentDerivativeKernel{Tdk,Tk<:Kernel} <: AbstractDerivativeKernel
    dk01::Tdk

    function SecondComponentDerivativeKernel(kernel)
        d01(t1, t2) = first(first(Zygote.gradient(t2 -> kernel(t1, t2), t2)))
        return new{typeof(d01),typeof(kernel)}(d01)
    end
end

(dk::SecondComponentDerivativeKernel)(t1, t2) = dk.dk01(t1, t2)

struct BothComponentDerivativeKernel{Tdk,Tk<:Kernel} <: AbstractDerivativeKernel
    dk11::Tdk

    function BothComponentDerivativeKernel(kernel)
        d11(t1, t2) = first(Zygote.hessian(t -> kernel(t[1], t[2]), [t1, t2])[2])
        return new{typeof(d11),typeof(kernel)}(d11)
    end
end

(dk::BothComponentDerivativeKernel)(t1, t2) = dk.dk11(t1, t2)

struct DerivativeKernelCollection{
    Tk<:Kernel,
    Tdk10<:FirstComponentDerivativeKernel{<:Function,Tk},
    Tdk01<:SecondComponentDerivativeKernel{<:Function,Tk},
    Tdk11<:BothComponentDerivativeKernel{<:Function,Tk},
}
    d10::Tdk10
    d01::Tdk01
    d11::Tdk11

    function DerivativeKernelCollection(kernel)
        d10 = FirstComponentDerivativeKernel(kernel)
        d01 = SecondComponentDerivativeKernel(kernel)
        d11 = BothComponentDerivativeKernel(kernel)
        return new{typeof(kernel),typeof(d10),typeof(d01),typeof(d11)}(d10, d01, d11)
    end
end

#  perhaps not super clear, but this is probably the most often needed part
function (dk::DerivativeKernelCollection)(t1, t2)
    return dk.d10(t1, t2)
end

# need tests, for various kernels and maybe a finitediff method to compare?