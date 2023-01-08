export gp_negloglikelihood

"""
    Returns two function function. 1) to compute the loglikelihood and 2) rebuild the GP from a new set of parameters. 

```julia
    gp_negloglikelihood(gp, x, y)
```

Arguments:

- `gp`: An AbstractGP.FiniteGP
- `x`: input data
- `y`: output data
"""
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