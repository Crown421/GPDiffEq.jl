export PULLEuler, PULLExpEuler
export ConstantBuffer, AdaptiveBuffer, FullBuffer

import SciMLBase: build_solution, ReturnCode, isadaptive

# Buffer object needed for all PULL solvers
abstract type AbstractBuffer end
struct FullBuffer <: AbstractBuffer end
struct ConstantBuffer <: AbstractBuffer
    size::Int
end
struct AdaptiveBuffer <: AbstractBuffer
    size::Int
    tol::Float64

    # might want keyword argument
    function AdaptiveBuffer(size, tol)
        if tol <= 0.0
            throw(ArgumentError("tol must be positive"))
        end
        return new(size, tol)
    end
end

isadaptive(alg::AbstractPULLAlg) = false

struct PULLEuler{B<:AbstractBuffer} <: AbstractPULLAlg
    buffer::B
end

struct PULLExpEuler{B<:AbstractBuffer} <: AbstractPULLAlg
    buffer::B
end

# ToDo: Automatic buffersize, maybe with constant/ adaptive buffer types?
function PULLEuler(; buffer=FullBuffer())
    return PULLEuler(buffer)
end

function PULLExpEuler(; buffer=FullBuffer())
    return PULLExpEuler(buffer)
end

# ToDo: not sure where this comes in
alg_order(alg::PULLEuler) = 1

function SciMLBase.__solve(prob::AbstractGPODEProblem, alg::AbstractPULLAlg; kwargs...)
    if :dt ∉ keys(kwargs)
        throw(ArgumentError("Must specify dt"))
    end
    dt = kwargs[:dt]

    gp = prob.f.gp
    dgp = prob.f.dgp

    nsteps = floor(Int, (prob.tspan[end] - prob.tspan[1]) / dt)

    ts = cumsum(vcat([prob.tspan[1]], dt * ones(nsteps)))
    finalstepdiff = prob.tspan[end] - ts[end]
    if finalstepdiff < 0.1 * dt
        ts[end] = prob.tspan[end]
    else
        push!(ts, prob.tspan[end])
    end

    # ToDo: get the type properly from prob uType
    xe = _pulleulerSolve(prob.u0, gp, dgp, ts, alg)

    return build_solution(prob, alg, ts, xe; retcode=ReturnCode.Success)
end

function _pulleulerSolve(u0, gp, dgp, ts, alg)
    _u0 = _mod_u0(u0)

    xe, av = _init(_u0, length(ts))

    tsteps = diff(ts)
    for (i, tstep) in enumerate(tsteps)
        step!(gp, dgp, xe, av, tstep, i, alg)
    end

    return _mod_out(xe, u0)
end

function _mod_out(xe, u0::Union{Real,Normal{<:Real}})
    return Normal.(only.(getfield.(xe, :μ)), sqrt.(only.(getfield.(xe, :Σ))))
end
_mod_out(xe, u0) = xe

_mod_u0(u0::Real) = MvNormal([u0], eps(eltype(u0)) * I)
_mod_u0(u0::AbstractVector{<:Real}) = MvNormal(u0, eps(eltype(u0)) * I)
# ToDo: Add tests for all of these, to make sure they do the right thing. In case of interface changes. 
_mod_u0(u0::Normal) = MvNormal([u0.μ], [u0.σ])
_mod_u0(u0::MvNormal) = u0

function _init(u0, nsteps)
    T = eltype(u0)
    xe = Vector{AbstractGPs.MvNormal{T}}(undef, nsteps)
    # av = [zeros(T, length(u0), length(u0)) for _ in 1:(nsteps)]
    d = length(u0)
    av = repeat(0.5 * Matrix{Float64}(I, d, d), 1, nsteps + 1)

    # Not the most elegant, but should work. 
    xe[1] = u0
    return xe, av
end

function _compute_coeffs(a, h, n, buffer::ConstantBuffer)
    lhist = buffer.size
    ldx = max(1, n - lhist)
    ah = ([I] .+ a[(ldx + 1):n] .* h)
    # println(ah)
    ahe = cumprod(reverse(ah))

    ahr = reduce(hcat, ahe; init=0.5 * I)
    return ldx, ahr
end

function _compute_coeffs(a, h, n, buffer::FullBuffer)
    ah = ([I] .+ a[2:n] .* h)
    ahe = cumprod(reverse(ah))

    ahr = reduce(hcat, ahe; init=0.5 * I)
    return 1, ahr
end

# function _compute_coeffs(a, n, buffer::AdaptiveBuffer)
#     lhist = buffer.size
#     ldx = max(1, n - lhist)
#     ah = ([I] .+ a[(ldx + 1):n] .* h)
#     ahe = cumprod(reverse(ah))

#     ahr = reduce(hcat, ahe; init=0.5 * I)
#     return ldx, ahr
# end

function step!(gp, dgp, x, a, h, n, alg::PULLEuler)
    μn = x[n].μ
    Σn = x[n].Σ
    # ToDo: Try refactoring derivative kernels, use `with_gradient` to get both faster

    an = I + h * _mean(dgp, μn)

    m = μn + h * _mean(gp, μn)

    # ldx, ahr = _compute_coeffs(a, h, n, buffer)
    ldx = 1
    d = length(μn)
    idxn = ((n - 1) * d + 1):(n * d)
    a[:, idxn] = an

    idx_1nm1 = (1:((n - 1) * d))
    @views a[:, idx_1nm1] = an * a[:, idx_1nm1]

    ahr = view(a, :, (d + 1):((n + 1) * d))
    μo = getfield.(x[ldx:n], :μ)

    Kr = _cov(gp, μo, [μo[end]])

    cv1 = ahr * Kr
    v1 = Symmetric(an * Σn * an')
    v = v1 + h^2 * (cv1 + cv1')

    return x[n + 1] = AbstractGPs.MvNormal(m, v)
end
