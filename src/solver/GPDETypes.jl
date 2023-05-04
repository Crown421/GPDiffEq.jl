import SciMLBase: AbstractODEFunction

"""
$(TYPEDEF)
"""
abstract type AbstractGPODEFunction{iip} <: AbstractODEFunction{iip} end

# @doc """

# Let's see what we need
# - thoughts:
#     - f is GP
#     - jac (kind of?) id DerivGP, might need some convenience function to get jacobian of mean. Maybe that happens when constructing a problem, based on the solver? or, wait, I will need 
# """
struct GPODEFunction{iip,GP,DGP} <: AbstractGPODEFunction{iip}
    gp::GP
    dgp::DGP
    # jac

    function GPODEFunction{iip}(
        gp::AbstractGPs.AbstractGP, dgp::AbstractGPs.AbstractGP
    ) where {iip}
        return new{iip,typeof(gp),typeof(dgp)}(gp, dgp)
    end
end

function GPODEFunction{iip}(gp::AbstractGPs.AbstractGP) where {iip}
    dgp = differentiate(gp)
    return GPODEFunction{iip}(gp, dgp)
end

# ToDo: something nice for inplace (detection)
function GPODEFunction(gp::AbstractGPs.AbstractGP)
    return GPODEFunction{false}(gp)
end

## ToDo: paramters at some point
# (gpf::GPODEFunction)(x, p, t) = _mean(gpf.gp, x)
(gpf::GPODEFunction)(x) = _mean(gpf.gp, x)

## GPDEProblem

import SciMLBase: promote_tspan, AbstractODEProblem, NullParameters
import DiffEqBase: get_concrete_u0, isconcreteu0

# later <: SciMLBase.AbstractDEProblem?
abstract type AbstractGPODEProblem{uType,tType,isinplace} <:
              AbstractODEProblem{uType,tType,isinplace} end

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
struct GPODEProblem{uType,tType,isinplace,P,GPFun,K} <:
       AbstractGPODEProblem{uType,tType,isinplace}
    f::GPFun
    u0::uType
    tspan::tType
    p::P  # for now no Parameters
    kwargs::K

    # @kw_only
    # https://github.com/SciML/SciMLBase.jl/blob/master/src/problems/sde_problems.jl
    function GPODEProblem{iip}(
        gp::AbstractGPODEFunction, u0, tspan, p, args...; kwargs...
    ) where {iip}
        _tspan = promote_tspan(tspan)
        return new{
            typeof(u0),typeof(_tspan),isinplace(gp),typeof(p),typeof(gp),typeof(kwargs)
        }(
            gp, u0, _tspan, p, kwargs
        )
    end
    # function GPODEProblem{iip}(gp, u0, tspan; kwargs...) where {iip}
    #     return GPODEProblem(GPFunction{iip}(gp), u0, tspan; kwargs...)
    # end
end

function get_concrete_u0(
    prob::GPODEProblem{<:Union{Normal,AbstractMvNormal}}, isadapt, t0, kwargs
)
    if haskey(kwargs, :u0)
        u0 = kwargs[:u0]
    else
        u0 = prob.u0
    end
    return u0
end

function isconcreteu0(prob::GPODEProblem{<:Union{Normal,AbstractMvNormal}}, t0, kwargs)
    return DiffEqBase.isdistribution(prob.u0)
end

"""
    GPODEProblem(f::ODEFunction,u0,tspan,p=NullParameters(),callback=CallbackSet())
Define an GPODE problem from an ... 
"""
function GPODEProblem(
    f::AbstractGPODEFunction, u0, tspan, p=NullParameters(), args...; kwargs...
)
    # ToDo: default false for now
    return GPODEProblem{false}(f, u0, tspan, p, args...; kwargs...)
end

# ToDo: printing?
