
import SciMLBase: promote_tspan

export GPODEProblem

# later <: SciMLBase.AbstractDEProblem?
abstract type AbstractGPODEProblem{uType,tType,isinplace} end

"""
Defines a Gaussian Process differential equation problem.

## Problem Type
In a GPODE problem, we assume that in the ODE
```math
\\dot{u}(t) = f(u(t), t, p)
```
the function ``f`` is given by a Gaussian process. 

ToDo: More details on GP interface and options (Regular, sparse, ...)

## Fields

"""
struct GPODEProblem{uType,tType,isinplace,GPFun,K} <:
       AbstractGPODEProblem{uType,tType,isinplace}
    f::GPFun
    u0::uType
    tspan::tType
    # p::P  # for now no Parameters
    K::K

    # @kw_only
    # https://github.com/SciML/SciMLBase.jl/blob/master/src/problems/sde_problems.jl
    function GPODEProblem{iip}(gp::AbstractGPODEFunction, u0, tspan; kwargs...) where {iip}
        _tspan = promote_tspan(tspan)
        return new{typeof(u0),typeof(_tspan),isinplace(gp),typeof(gp),typeof(kwargs)}(
            gp, u0, _tspan, kwargs
        )
    end
    # function GPODEProblem{iip}(gp, u0, tspan; kwargs...) where {iip}
    #     return GPODEProblem(GPFunction{iip}(gp), u0, tspan; kwargs...)
    # end
end

"""
    ODEProblem(f::ODEFunction,u0,tspan,p=NullParameters(),callback=CallbackSet())
Define an ODE problem from an [`ODEFunction`](@ref).
"""
function GPODEProblem(f::AbstractGPODEFunction, u0, tspan, args...; kwargs...)
    # ToDo: default false for now
    return GPODEProblem{false}(f, u0, tspan, args...; kwargs...)
end

# ToDo: printing?
