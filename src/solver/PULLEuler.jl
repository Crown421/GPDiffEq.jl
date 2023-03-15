export PULLEuler

import SciMLBase: build_solution, ReturnCode

# Buffer object needed for all PULL solvers

struct PULLEuler <: AbstractPULLAlg
    buffersize::Int
end

# ToDo: Automatic buffersize, maybe with constant/ adaptive buffer types?
function PULLEuler(; buffersize=100)
    return PULLEuler(buffersize)
end

# ToDo: not sure where this comes in
alg_order(alg::PULLEuler) = 1

# ToDo: Split out subfunction, dispatching scalars vs vector. 
# For the scalar version, make into 1D vector, call vector version, and then convert MvNormal to normal.
# Also takes care of the _init issue. 
function SciMLBase.__solve(prob::AbstractGPODEProblem, alg::PULLEuler; dt, kwargs...)
    gp = prob.f.gp
    dgp = prob.f.dgp
    # dt = kwargs[:dt]

    nsteps = ceil(Int, (prob.tspan[end] - prob.tspan[1]) / dt)

    # ToDo: get the type properly from prob uType
    ts, xe = _pulleulerSolve(prob.u0, nsteps, gp, dgp, prob.tspan, dt, alg)

    return build_solution(prob, alg, ts, xe; retcode=ReturnCode.Success)
end

function _pulleulerSolve(
    u0::AbstractVector{T}, nsteps, gp, dgp, tspan, dt, alg
) where {T<:Real}
    xe, av = _init(u0, nsteps)

    for i in 1:nsteps
        linearized_eulerstep!(gp, dgp, xe, av, dt, i; lhist=alg.buffersize)
    end
    ts = tspan[1]:dt:(dt * nsteps)

    return ts, xe
end

function _pulleulerSolve(u0::T, nsteps, gp, dgp, tspan, dt, alg) where {T<:Real}
    ts, xe = _pulleulerSolve([u0], nsteps, gp, dgp, tspan, dt, alg)
    xes = Normal.(only.(getfield.(xe, :μ)), sqrt.(only.(getfield.(xe, :Σ))))
    return ts, xes
end

# ToDo: move to utils or do better
function _init(u0::T, nsteps) where {T<:Real}
    xe = zeros(Measurement{T}, nsteps + 1)
    av = zeros(T, nsteps + 1)
    xe[1] = u0
    return xe, av
end

function _init(u0::AbstractVector{T}, nsteps) where {T<:Real}
    xe = Vector{AbstractGPs.MvNormal{T}}(undef, nsteps + 1)
    av = [zeros(T, length(u0), length(u0)) for _ in 1:(nsteps + 1)]
    # Not the most elegant, but should work. 
    xe[1] = AbstractGPs.MvNormal(u0, eps(T) * I)
    return xe, av
end

function quadf2(ahn, Σ)
    C = ahn * Σ * ahn'
    return 0.5 * (C + C')
end

function linearized_eulerstep!(gp, dgp, x, a, h, n; lhist=150)
    # ToDo: fix / add sugar for abstract vector requirement of mean when only using scalar
    # ToDo: rename to mean and variance related names
    xnval = x[n].μ
    xnerr = x[n].Σ
    a[n] .= _mean(dgp, xnval) # this needs the derivative

    m = xnval + h * _mean(gp, xnval)

    ldx = max(1, n - lhist)
    μo = getfield.(x[ldx:n], :μ)

    Kr = _cov(gp, reverse(μo), [μo[end]])
    ah = ([I] .+ a[(ldx + 1):n] .* h)
    # ahe = reverse(cumprod(reverse(ah)))
    ahe = cumprod(reverse(ah))

    # ToDo: move up, use ahe to determine ldx automatically
    ahr = reduce(hcat, ahe; init=0.5 * I)

    # println(Kr)
    # println(ahr)

    cv1 = ahr * Kr
    # really wonder if this is correct, and if so, if I can just make it better with A + A'?
    cv2 = Kr' * ahr'
    # (I + a[n] * h) * xnerr * (I + a[n] * h)'
    v = quadf2((I + a[n] * h), xnerr) + h^2 * (cv1 + cv2)

    # # println(v)
    # println("v: $(det(v))")
    # println("a: $(det(a[n] * h))")

    return x[n + 1] = AbstractGPs.MvNormal(m, v)
    #m .± sqrt.(diag(v))
end

# TODO: !!!!!!
# For independent kernel, somehow make it `DiagNormal`

# ToDo: need to seriously adjust the `_mean` function. Perhaps an initial hack would be to specialize on DerivGP and Vector input?
