export PULLEuler

struct PULLEuler <: AbstractPULLAlg end

alg_order(alg::PULLEuler) = 1

# @cache 
struct PULLEulerCache{uType,rateType,StageLimiter,StepLimiter,TabType} <:
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

struct PULLEulerConstantCache{T,T2} <: OrdinaryDiffEqConstantCache end

function PULLEuler(T, T2) end

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
) end

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
    integrator.kshortsize = 2
    integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)
    integrator.fsalfirst = integrator.f(integrator.uprev, integrator.p, integrator.t) # Pre-start fsal
    integrator.destats.nf += 1

    # Avoid undefined entries if k is an array of arrays
    integrator.fsallast = zero(integrator.fsalfirst)
    integrator.k[1] = integrator.fsalfirst
    return integrator.k[2] = integrator.fsallast
end

function perform_step!(integrator, cache::PULLEulerConstantCache, repeat_step=false)
    @unpack t, dt, uprev, f, p = integrator
    @muladd u = @.. broadcast = false uprev + dt * integrator.fsalfirst
    k = f(u, p, t + dt) # For the interpolation, needs k at the updated point
    integrator.destats.nf += 1
    integrator.fsallast = k
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    return integrator.u = u
end

function initialize!(integrator, cache::PULLEulerCache)
    integrator.kshortsize = 2
    @unpack k, fsalfirst = cache
    integrator.fsalfirst = fsalfirst
    integrator.fsallast = k
    resize!(integrator.k, integrator.kshortsize)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast
    integrator.f(integrator.fsalfirst, integrator.uprev, integrator.p, integrator.t) # For the interpolation, needs k at the updated point
    return integrator.destats.nf += 1
end

function perform_step!(integrator, cache::PULLEulerCache, repeat_step=false)
    @unpack t, dt, uprev, u, f, p = integrator
    @muladd @.. broadcast = false u = uprev + dt * integrator.fsalfirst
    f(integrator.fsallast, u, p, t + dt) # For the interpolation, needs k at the updated point
    return integrator.destats.nf += 1
end