
import SciMLBase: promote_tspan

# later <: SciMLBase.AbstractDEProblem?
abstract type AbstractGPDEProblem{uType,tType,isinplace} end

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
struct GPDEProblem{uType,tType,isinplace,GPFun,K} <:
       AbstractGPDEProblem{uType,tType,isinplace}
    f::GPFun
    u0::uType
    tspan::tType
    # p::P  # for now no Parameters
    K::K

    # @kw_only
    # https://github.com/SciML/SciMLBase.jl/blob/master/src/problems/sde_problems.jl
    function GPDEProblem{iip}(gp::AbstractGPFunction, u0, tspan; kwargs...) where {iip}
        _tspan = promote_tspan(tspan)
        return new{typeof(u0),typeof(_tspan),isinplace(gp),typeof(f),typeof(kwargs)}(
            gp, u0, _tspan, p, kwargs
        )
    end
    function GPDEProblem{iip}(gp, u0, tspan; kwargs...) where {iip}
        return GPDEProblem(GPFunction{iip}(gp), u0, tspan; kwargs...)
    end
end
