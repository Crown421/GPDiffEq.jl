export PULLEuler

import SciMLBase: build_solution

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
    buffer = alg.buffersize

    nsteps = ceil(Int, (prob.tspan[end] - prob.tspan[1]) / dt)

    # ToDo: get the type properly from prob uType
    xe, av = _init(prob.u0, nsteps)

    for i in 1:nsteps
        linearized_eulerstep!(gp, dgp, xe, av, dt, i; lhist=alg.buffersize)
    end
    ts = prob.tspan[1]:dt:(dt * nsteps)

    return build_solution(prob, alg, ts, xe; retcode=:Success)
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
# change the output type from Measurement to Distributions.MvNormal. Once again have to see how that plays with the scalar case. Could also just transform scalar input to 1D vector and transform back later. 
# For independent kernel, somehow make it `DiagNormal`

# ToDo: need to seriously adjust the `_mean` function. Perhaps an initial hack would be to specialize on DerivGP and Vector input?

# need to fix random see for sampling tests

### random old stuff
# d = length(xnval)

# # map(x -> getfield.(x, :val), x[ldx:n])
# # getfield.(x[ldx:n], :val)
# # K = vtmp(μo)
# # ToDo: Swapping to vectors everywhere, I probably don't need this wrapper anymore
# K = _cov(gp, μo)
# Kbr = K[(end - (d - 1)):end, :]

# # vb = Kbr[(end - (d - 1)):end, (end - (d - 1)):end]

# # something likel
# # `prod.([collect(1:10)[end-1:-1:end-i] for i in 0:5])`
# # could be faster/ better/ more readable?
# ah = ([I] .+ a[(ldx + 1):n] .* h)
# ahe = reverse(cumprod(reverse(ah)))

# println(ahe)
# println(Kbr[:, 1:(end - d)])

# # cv1 = reduce(vcat, ahe)' * Kbr[:, 1:(end - d)]'
# cv2 = Kbr[:, 1:(end - d)] * reduce(vcat, adjoint.(ahe))
# # ToDo: Find a better solution
# cv = isempty(cv) ? zeros(eltype(cv), d, d) : cv

# v = (I + a[n] * h) * xnerr * (I + a[n] * h)' + h^2 * vb + h^2 * (cv1 + cv2)
# display(v)

function old_solve(prob::AbstractGPODEProblem, alg::PULLEuler; dt, kwargs...)
    gp = prob.f.gp
    dgp = prob.f.dgp
    # dt = kwargs[:dt]
    buffer = alg.buffersize

    nsteps = ceil(Int, (prob.tspan[end] - prob.tspan[1]) / dt)

    # ToDo: get the type properly from prob uType
    xe = zeros(Measurement{Float64}, nsteps + 1)
    av = zeros(nsteps + 1)

    # ToDo: check lift to u0 ± 0 when only scalar is given (might already be happening)
    xe[1] = prob.u0 # μ₀ ± sqrt(Σ₀)

    for i in 1:nsteps
        old_linearized_eulerstep!(gp, dgp, xe, av, dt, i; lhist=alg.buffersize)
    end
    ts = prob.tspan[1]:dt:(dt * nsteps)

    return build_solution(prob, alg, ts, xe; retcode=:Success)
end

function old_linearized_eulerstep!(gp, dgp, x, a, h, n; lhist=150)
    # ToDo: fix / add sugar for abstract vector requirement of mean when only using scalar
    a[n] = _mean(dgp, x[n].val) # this needs the derivative

    m = x[n].val + h * _mean(gp, x[n].val)

    ldx = max(1, n - lhist)
    μo = getfield.(x[ldx:n], :val)

    # K = vtmp(μo)
    K = cov(gp, μo)
    Kbr = K[end, :]
    vb = Kbr[end]

    # ToDo: move up, use ahe to determine ldx automatically
    ah = (1 .+ a[(ldx + 1):(n - 1)] .* h)
    ahe = reverse(vcat(1.0, cumprod(reverse(ah))))

    cv = sum(ahe .* Kbr[1:(end - 1)])
    v = (1 + a[n] * h)^2 * x[n].err^2 + h^2 * vb + 2 * h^2 * cv * (1 + a[n] * h)

    return x[n + 1] = m ± sqrt(v)
end