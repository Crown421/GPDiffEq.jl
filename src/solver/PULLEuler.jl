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
    xe = [zeros(Measurement{T}, length(u0)) for _ in 1:(nsteps + 1)]
    av = [zeros(T, length(u0), length(u0)) for _ in 1:(nsteps + 1)]
    xe[1] = u0
    return xe, av
end

function linearized_eulerstep!(gp, dgp, x, a, h, n; lhist=150)
    # ToDo: fix / add sugar for abstract vector requirement of mean when only using scalar
    xnval = getfield.(x[n], :val)
    xnerr = getfield.(x[n], :err)
    d = length(xnval)
    a[n] = _mean(dgp, xnval) # this needs the derivative

    m = xnval + h * _mean(gp, xnval)

    ldx = max(1, n - lhist)
    μo = map(x -> getfield.(x, :val), x[ldx:n])
    # getfield.(x[ldx:n], :val)

    # K = vtmp(μo)
    K = _cov(gp, μo)
    Kbr = K[(end - (d - 1)):end, :]

    vb = Kbr[(end - (d - 1)):end, (end - (d - 1)):end]

    # ToDo: move up, use ahe to determine ldx automatically
    ah = ([I] .+ a[(ldx + 1):(n - 1)] .* h)
    ahe = reverse(vcat([I], cumprod(reverse(ah))))

    cv = reduce(vcat, ahe)' * Kbr[:, 1:(end - d)]'
    # ToDo: Find a better solution
    cv = isempty(cv) ? zeros(eltype(cv), d, d) : cv

    v =
        (I + a[n] * h) * diagm(xnerr .^ 2) * (I + a[n] * h)' +
        h^2 * vb +
        2 * h^2 * (I + a[n] * h) * cv
    display(v)

    return x[n + 1] = m .± sqrt.(diag(v))
end

# TODO: !!!!!!
# change the output type from Measurement to Distributions.MvNormal. Once again have to see how that plays with the scalar case. Could also just transform scalar input to 1D vector and transform back later. 
# For independent kernel, somehow make it `DiagNormal`

# how to 

# ToDo: order of `cv` inner product might matter? Kbr is symmetric, but then later when multiplying with (1+ah)? Need to check this

# ToDo: need to seriously adjust the `_mean` function. Perhaps an initial hack would be to specialize on DerivGP and Vector input?

# need to fix random see for sampling tests