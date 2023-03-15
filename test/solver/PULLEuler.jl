using GPDiffEq
using Random
using Test

# optional dependency
using DifferentialEquations

Random.seed!(1234)

## Comparison implementation

function old_solve(prob, alg::PULLEuler; dt, kwargs...)
    gp = prob.f.gp
    dgp = prob.f.dgp
    # dt = kwargs[:dt]

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

    return DifferentialEquations.SciMLBase.build_solution(
        prob, alg, ts, xe; retcode=DifferentialEquations.SciMLBase.ReturnCode.Success
    )
end

function old_linearized_eulerstep!(gp, dgp, x, a, h, n; lhist=150)
    # ToDo: fix / add sugar for abstract vector requirement of mean when only using scalar
    a[n] = GPDiffEq.PullSolversModule._mean(dgp, x[n].val) # this needs the derivative

    m = x[n].val + h * GPDiffEq.PullSolversModule._mean(gp, x[n].val)

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

@testset "Scalar" begin
    f(x) = x * cos(x)

    X = range(-3.0, 3.0; length=10)
    σ_n = 0.1
    y = f.(X) .+ σ_n * randn(length(X))

    ker = SqExponentialKernel()
    gp = GP(ker)
    fx = gp(X, σ_n^2)

    fp = posterior(fx, y)

    # u0 = [1.0]
    u0 = 1.0
    tspan = (0.0, 8.0)
    ff = GPODEFunction(fp)
    prob = GPODEProblem(ff, u0, tspan)

    # and integrate with the PULL Euler solver. 
    sol = solve(prob, PULLEuler(); dt=0.1)

    # need to make scalars work
    refsol = old_solve(prob, PULLEuler(); dt=0.1)

    @test getfield.(refsol.u, :err) ≈ getfield.(sol.u, :σ) atol = 1e-6
    @test getfield.(refsol.u, :val) ≈ getfield.(sol.u, :μ)

    refdetsol = solve(prob, Euler(); dt=0.1)
    @test getfield.(sol.u, :μ) ≈ refdetsol.u
end

function fun(x)
    return [-0.1 2.0; -2.0 -0.1] * (x .^ 3)
end
xrange = range(-2.2, 2.2; length=6)
x = collect.(Iterators.product(xrange, xrange))[:]
y = fun.(x)
## This is annoying UX, needs fix
y = ColVecs(reduce(hcat, y))
xMO, yMO = prepare_isotopic_multi_output_data(x, y)

# ### Defining a Multi-Output GP
σ_n = 3e-2
ker = SqExponentialKernel()
mker = IndependentMOKernel(ker)

gp = GP(mker)
fx = gp(xMO, σ_n)
fp = posterior(fx, yMO)

# Define a `GPODEProblem`
h = 0.002
u0 = [2.0; 0.0]
tspan = (0.0, 4.0)
gpff = GPODEFunction(fp)

prob = GPODEProblem(gpff, u0, tspan)

refdetsol = solve(prob, Euler(); dt=h)

sol = solve(prob, PULLEuler(); dt=h)

@test getfield.(sol.u, :μ) ≈ refdetsol.u atol = 1e-3