# ToDo: nicer implementation
struct FirstCompDerivKernel{K} <: Kernel
    onegrad::Function

    function FirstCompDerivKernel(kernel)
        dker(t1, t2) = first(first(gradient(t1 -> kernel(t1, t2), t1)))
        return new{typeof(kernel)}(dker)
    end
end

(dk::FirstCompDerivKernel)(x, y) = dk.onegrad(x, y)

function build_deriv_model(gp::AbstractGPs.PosteriorGP)
    kernel = FirstCompDerivKernel(gp.prior.kernel)
    f = GP(kernel)
    # ToDo: This is brittle, needs to take into account mean
    return AbstractGPs.PosteriorGP(f, gp.data)
end