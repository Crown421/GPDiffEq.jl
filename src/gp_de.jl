export GPODE

abstract type AbstractGPDE <: Function end
basic_tgrad(u, p, t) = zero(u)
# Flux.trainable

struct GPODE{M,T,A,K} <: AbstractGPDE
    model::M # AbstractGP
    tspan::T
    args::A
    kwargs::K

    function GPODE(model, tspan, args...; p=nothing, kwargs...)
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