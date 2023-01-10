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

function SciMLBase.__solve(prob::AbstractGPODEProblem, alg::PULLEuler; dt, kwargs...)
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
        linearized_eulerstep!(gp, dgp, xe, av, dt, i; lhist=alg.buffersize)
    end
    ts = prob.tspan[1]:dt:(dt * nsteps)

    return build_solution(prob, alg, ts, xe; retcode=:Success)
end

function linearized_eulerstep!(gp, dgp, x, a, h, n; lhist=150)
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