abstract type AbstractGPDE <: Function end

# Flux.trainable

struct GPODE{M,P,RC,T,A,K} <: AbstractGPDE
    model::M # AbstractGP
    p::P # kernel parameters
    rc::RC
    tspan::T
    args::A
    kwargs::K

    function GPODE(model, tspan, args...; p=nothing, kwargs...)
        # ToDo: Add destructure for model
        _p, rc = Flux.destructure(model)
        if p === nothing
            p = _p
        end
        return new{
            typeof(model),typeof(p),typeof(rc),typeof(tspan),typeof(args),typeof(kwargs)
        }(
            model, p, re, tspan, args, kwargs
        )
    end
end