module PullSoversModule

using OrdinaryDiffEq
import OrdinaryDiffEq:
    OrdinaryDiffEqAlgorithm,
    OrdinaryDiffEqMutableCache,
    OrdinaryDiffEqConstantCache,
    alg_order,
    alg_cache,
    initialize!,
    perform_step!,
    trivial_limiter!,
    constvalue,
    @muladd,
    @unpack,
    @cache,
    @..

struct PULLEuler <: OrdinaryDiffEqAlgorithm end

export PULLEuler

alg_order(alg::PULLEuler) = 1

@cache struct PULLEuler_ALGCache{uType,rateType,StageLimiter,StepLimiter,TabType} <:
              OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
    k::rateType
    tmp::uType
    u₂::uType
    fsalfirst::rateType
    stage_limiter!::StageLimiter
    step_limiter!::StepLimiter
    tab::TabType
end

struct PULLEuler{T,T2} <: OrdinaryDiffEqConstantCache
end

function PULLEuler(T, T2)
end

function alg_cache(
    alg::PULLEuler,
    u,
    rate_prototype,
    uEltypeNoUnits,
    uBottomEltypeNoUnits,
    tTypeNoUnits,
    uprev,
    uprev2,
    f,
    t,
    dt,
    reltol,
    p,
    calck,
    ::Val{true},
)
end

function alg_cache(
    alg::PULLEuler,
    u,
    rate_prototype,
    uEltypeNoUnits,
    uBottomEltypeNoUnits,
    tTypeNoUnits,
    uprev,
    uprev2,
    f,
    t,
    dt,
    reltol,
    p,
    calck,
    ::Val{false},
)
    return PULLEulerConstantCache(real(uBottomEltypeNoUnits), real(tTypeNoUnits))
end

function initialize!(integrator, cache::PULLEulerConstantCache)
end

@muladd function perform_step!(integrator, cache::PULLEulerConstantCache, repeat_step=false)
end

function initialize!(integrator, cache::PULLEulerCache)
end

@muladd function perform_step!(integrator, cache::PULLEulerCache, repeat_step=false)
end

end