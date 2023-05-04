# using GPDiffEq
# using DifferentialEquations
import SciMLBase.solve
import SciMLBase.EnsembleProblem

export SampledGPODEProblem, GPODESampledEnsembleProblem
export NoReduction, OnlineReduction, CustomReduction, FitGenNormal

_sample_input(u0::Union{Real,Vector{<:Real}}) = u0
_sample_input(u0::Union{Normal,MvNormal}) = rand(u0)

# later I can have multiple sample methods
struct SampledGPODEProblem{R,G,U}
    prob::ODEProblem
    range::R
    gp::G
    u0::U

    # xrange as argument. Could do some refinement of GP points grid as default?
    function SampledGPODEProblem(gpf::GPODEFunction, xrange, u0, tspan)
        fs = sample_function(gpf.gp, xrange)
        u0s = _sample_input(u0)

        sprob = ODEProblem((u, p, t) -> p(u), u0s, tspan, fs)
        # new{typeof(sprob)}(sprob)
        return new{typeof(xrange),typeof(gpf.gp),typeof(u0)}(sprob, xrange, gpf.gp, u0)
    end
end

# want to replace the first word of this one
# show(prob::SampledGPProblem) = show(prob.prob)
solve(prob::SampledGPODEProblem, args...; kwargs...) = solve(prob.prob, args...; kwargs...)

############## 
# EnsembleProblems

abstract type AbstractReduction end
struct NoReduction <: AbstractReduction end
# want to store the online stat here, and have this be the place where other OnlineStats can be added as well. Would be great to set a default, but, at this point I don't know the input?
# Possibly need a new type?
struct OnlineReduction{S} <: AbstractReduction
    stat::S
end
struct CustomReduction <: AbstractReduction
    func::Function
end

"""
    GPODESampledEnsembleProblem(prob::SampledGPODEProblem, nGPSamples, [nInValSamples]; kwargs...)
Defines a GPODESampledEnsembleProblem, which is a convenience wrapper around a SciML.EnsembleProblem (https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/) to compute the empirical distribution of the solution of a GPODEProblem. 

Arguments:
- `prob`: a SampledGPODEProblem
- `reduction`: an `AbstractReduction`, which defines how to reduce the ensemble of solutions. Default is OnlineReduction(FitGenNormal()), which returns a (Mv)Normal distribution for each state.
- `nGPSamples`: number of samples from the GPODEFunction
- `nInValSamples`: number of samples from the initial value distribution. Only needed if `u0` is a normal distribution (i.e. Normal or MvNormal).

Keyword arguments:
The normal keyword arguments for SciML.EnsembleProblems are supported, except for `prob_func` and `reduction`.
"""
struct GPODESampledEnsembleProblem{S<:AbstractReduction}
    prob::EnsembleProblem
    nInValSamples::Int
    nGPSamples::Int
    # prob_func::Function
    # reduction::AbstractReduction
    function GPODESampledEnsembleProblem(
        sgpp::SampledGPODEProblem,
        reduction::AbstractReduction=OnlineReduction(FitGenNormal());
        nGPSamples,
        nInValSamples=nothing,
        kwargs...,
    )
        nInValSamples = _check_kwargs(sgpp.u0, nInValSamples, kwargs)
        prob_func = _define_prob_func(
            sgpp.u0, nInValSamples, sgpp.gp, sgpp.range, nGPSamples
        )
        red = _define_reduction(sgpp.prob.u0, reduction)
        ensprob = EnsembleProblem(sgpp.prob; prob_func, reduction=red, kwargs...)
        return new{typeof(reduction)}(ensprob, nInValSamples, nGPSamples)
    end
end

# ToDo: Add more data
function Base.show(io::IO, ::MIME"text/plain", ensprob::GPODESampledEnsembleProblem)
    return print(
        io,
        nameof(typeof(ensprob)),
        " with uType ",
        typeof(ensprob.prob.prob.u0),
        "\n   reduction: ",
        typeof(ensprob).parameters[1],
        "\n   GP samples: ",
        ensprob.nGPSamples,
        if ensprob.nInValSamples > 1
            "\n   Initial value samples: $(ensprob.nInValSamples)"
        else
            ""
        end,
    )
end

# setup

function _check_kwargs(u0, nInValSamples, kwargs)
    if u0 isa Distribution #Union{Normal,MvNormal}
        if isnothing(nInValSamples)
            throw(ArgumentError("Must specify nInValSamples for a distributional u0."))
        end
    else
        if !isnothing(nInValSamples)
            @warn "Specifying nInValSamples for a concrete u0 has no effect"
        end
        nInValSamples = 1
    end
    if :prob_func ∈ keys(kwargs) || :reduction ∈ keys(kwargs)
        throw(ArgumentError("Cannot specify prob_func or reduction."))
    end
    return nInValSamples
end

## for u0 a scalar

function _define_prob_func(
    u0::Union{Real,Vector{<:Real}}, nInValSamples, gp, xrange, nGPSamples
)
    fv = [sample_function(gp, xrange) for _ in 1:nGPSamples]
    return (prob, i, repeat) -> _prob_func(prob, i, repeat, fv)
end

function _prob_func(prob, i, repeat, fv)
    return remake(prob; p=fv[i])
end

## for u0 a distribution

function _define_prob_func(
    u0d::Union{Normal,MvNormal}, nInValSamples, gp, xrange, nGPSamples
)
    u0v = [rand(u0d) for _ in 1:nInValSamples]
    fv = [sample_function(gp, xrange) for _ in 1:nGPSamples]
    return (prob, i, repeat) -> _prob_func(prob, i, repeat, u0v, fv, nInValSamples)
end

function _prob_func(prob, i, repeat, u0v, fv, nInValSamples)
    j = mod1(i, nInValSamples)
    k = fld1(i, nInValSamples)
    return remake(prob; u0=u0v[j], p=fv[k])
end

# function _prob_func(prob, i, repeat, u0d, xrange, gp)
#     s = rand(u0d)
#     fs = sample_function(gp, xrange)
#     return remake(prob; u0=s, p=fs)
# end

# function _define_prob_func(u0d, nInValSamples, xrange, gp)
#     # u0v = [rand(u0d) for _ in 1:nInValSamples]
#     return (prob, i, repeat) -> _prob_func(prob, i, repeat, u0d, nInValSamples, xrange, gp)
# end

### convenience wrapper for EnsembleProblem
# for "non-online" solving, returning full list of trajectories

## Reduction internals

function _define_reduction(u0, ::NoReduction)
    return SciMLBase.DEFAULT_REDUCTION
end

function _define_reduction(u0, cred::CustomReduction)
    return cred.func
end

function _define_reduction(u0, ored::OnlineReduction)
    ost = _onlinestat(u0, ored.stat)
    return (u, data, I) -> _reduction(u, data, I, ost)
end

# Combotype of Normal and MvNormal
# Ideally would be an AbstractOnlineStat, but that does not exist
struct FitGenNormal end

# need to pass type of `u0` in (once issue in OnlineStats.jl is addressed)
_onlinestat(u0::Real, ::FitGenNormal) = FitNormal()
_onlinestat(u0::AbstractVector, ::FitGenNormal) = FitMvNormal(length(u0))
_onlinestat(u0, ostat::OnlineStat) = ostat

function mfit!(os, iter)
    for yi in iter
        fit!(os, yi)
    end
    return os
end

function _reduction(u, data, I, ost)
    if isempty(u)
        base = data[1]
        m = [deepcopy(ost) for _ in 1:length(base.u)]
        u = [SciMLBase.build_solution(base.prob, base.alg, base.t, m)]
    end
    # tIdx = getindex.(getfield.(data, :u), length(u[1].u)) .< 1
    # print(" $(sum(tIdx))")
    sIdx = getfield.(data, :retcode) .== ReturnCode.Success
    mfit!.(u[1].u, eachrow(reduce(hcat, getfield.(data[sIdx], :u))))
    return (u, false)
end

function solve(prob::GPODESampledEnsembleProblem, alg, args...; kwargs...)
    if (:dt ∉ keys(kwargs)) && (:saveat ∉ keys(kwargs))
        throw(ArgumentError("Must specify either dt or saveat"))
    end
    nTraj = prob.nInValSamples * prob.nGPSamples
    if (:batch_size ∉ keys(kwargs))
        nb = min(1000, nTraj)
        kwargs = merge((; :batch_size => nb), kwargs)
    end

    if :trajectories ∈ keys(kwargs)
        throw(
            ArgumentError(
                "trajectories keyword specified by number of samples in GPODESampledEnsembleProblem",
            ),
        )
    end
    kwargs = merge((; :trajectories => nTraj), kwargs)
    return _solve(prob, alg, args...; kwargs...)
end

function _solve(
    prob::GPODESampledEnsembleProblem{<:OnlineReduction}, alg, args...; kwargs...
)
    return only(solve(prob.prob, alg, args...; kwargs...))
end

function _solve(prob::GPODESampledEnsembleProblem, alg, args...; kwargs...)
    return solve(prob.prob, alg, args...; kwargs...)
end

## wrapper type

# struct OnlineEnsembleProblem{U}
#     prob::EnsembleProblem
#     nInValSamples::Int
#     nGPSamples::Int

#     function OnlineEnsembleProblem(
#         sgpp::SampledGPODEProblem, trajostat; nGPSamples, kwargs...
#     )
#         nInValSamples = _check_nInValSamples(sgpp.u0, kwargs...)

#         reduction = _define_reduction(sgpp.prob.u0, trajostat)
#         prob_func = _define_prob_func(u0d, nInValSamples, sgpp.range, sgpp.gp, nGPSamples)
#         ensprob = EnsembleProblem(sgpp.prob; prob_func, reduction, kwargs...)
#         return new{typeof(sgpp.u0)}(ensprob, nInValSamples, nGPSamples)
#     end
# end

# maybe instead of dispatching on u0, do SampledGPODEProblem{R,G, <:Distribution} where {R,G}
# function _defineOnlineEnsembleProblem(
#     sgpp::SampledGPODEProblem,
#     trajostat,
#     u0::Union{Real,Vector{<:Real}};
#     nGPSamples=100,
#     kwargs...,
# )
#     reduction = _define_reduction(sgpp.prob.u0, trajostat)
#     prob_func = _define_prob_func(u0, nInValSamples, sgpp.range, sgpp.gp, nGPSamples)
#     ensprob = EnsembleProblem(sgpp.prob; prob_func, reduction, kwargs...)
#     return ensprob, 1
# end

# function _defineOnlineEnsembleProblem(
#     sgpp::SampledGPODEProblem, trajostat, u0d::Union{Normal,MvNormal}; nGPSamples, kwargs...
# )
#     if u0d isa Union{Normal,MvNormal}
#         if :nInValSamples ∉ keys(kwargs)
#             throw(ArgumentError("Must specify nInValSamples for a distributional u0"))
#         end
#         nInValSamples = kwargs[:nInValSamples]
#     else
#         nInValSamples = 1
#     end
#     reduction = _define_reduction(sgpp.prob.u0, trajostat)
#     prob_func = _define_prob_func(u0d, nInValSamples, sgpp.range, sgpp.gp, nGPSamples)
#     ensprob = EnsembleProblem(sgpp.prob; prob_func, reduction, kwargs...)
#     return ensprob, nInValSamples
# end