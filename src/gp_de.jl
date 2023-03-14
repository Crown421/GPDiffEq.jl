########################
## Old Implementation

abstract type AbstractGPDE <: Function end
basic_tgrad(u, p, t) = zero(u)
# Flux.trainable

"""
Defines an ODE model with a GP vector field. The returned model can be evaluated for an initial value as input.

```julia
    GPODE(model, tspan, args...; kwargs...)
```

Arguments:

- `model`: An AbstractGP.FiniteGP
- `tspan`: The time span for the ODE solver
- `args`: Additional positional arguments from DifferentialEquations.jl
- `kwargs`: Additional keyword arguments from DifferentialEquations.jl
"""
struct GPODE{M,T,A,K} <: AbstractGPDE
    model::M # AbstractGP
    tspan::T
    args::A
    kwargs::K

    function GPODE(model, tspan, args...; kwargs...)
        return new{typeof(model),typeof(tspan),typeof(args),typeof(kwargs)}(
            model, tspan, args, kwargs
        )
    end
end

function (gpode::GPODE)(u0)
    function dudt_(u, p, t)
        u_mo = MOInput([u], length(u))
        return mean(gpode.model, u_mo)
    end
    ff = ODEFunction{false}(dudt_; tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff, u0, getfield(gpode, :tspan))
    return solve(prob, gpode.args...; gpode.kwargs...)
end