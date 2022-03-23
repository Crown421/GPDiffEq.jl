export gp_negloglikelihood

# ToDo: type annotation?AbstractGP
function gp_negloglikelihood(gp, x, y)
    return _gp_negloglikelihood(gp, x, y)
end

function _gp_negloglikelihood(gp::AbstractGPs.FiniteGP, x, y)
    _p, kernelrc = Flux.destructure(gp.f.kernel)
    # ToDo: noise optimization
    σ_n = first(gp.Σy)
    function loglikelihood(params)
        kernel = kernelrc(params)
        f = GP(kernel)
        #ToDo: work on this
        fx = f(gp.x, σ_n)
        return -logpdf(fx, y)
    end
    function buildgppost(params)
        kernel = kernelrc(params)
        f = GP(kernel)
        #ToDo: work on this
        fx = f(gp.x, σ_n)
        return posterior(fx, y)
    end
    return loglikelihood, buildgppost
end

# one for dlc Approx/ VLA

## Piracy, need to fix in KernelFunctions (Issue)
Flux.@functor IndependentMOKernel
# import Flux: destructure
# Flux.destructure()
