
using Test
using Random
using GPDiffEq
using LinearAlgebra

Random.seed!(1234)

@testset "Scalar" begin
    s = range(-4.25, 4.252; length=100)
    f(x) = x * cos(x)

    X = range(-3.0, 3.0; length=10)
    σ_n = 0.1
    y = f.(X) .+ σ_n * randn(length(X))

    ker = SqExponentialKernel()
    gp = GP(ker)
    fx = gp(X, σ_n^2)

    fp = posterior(fx, y)

    # GPODE
    tspan = (0.0, 8.0)
    h = 0.05
    gpf = GPODEFunction(fp)
    xrange = range(0.0, 2.2; length=6)

    @testset "Concrete initial value" begin
        u0 = 1.0

        sprob = SampledGPODEProblem(gpf, xrange, u0, tspan)

        # testing single sample
        @test sprob isa SampledGPODEProblem
        @test sprob.prob.p isa
            GPDiffEq.PullSolversModule.Interpolations.AbstractInterpolationWrapper

        ssol = solve(sprob, Euler(); dt=h)

        @test ssol isa ODESolution

        # "normal" ensemble

        ntraj = 1000
        ensprob = GPODESampledEnsembleProblem(sprob, NoReduction(); nGPSamples=ntraj)

        @test_logs (:warn, "Specifying nInValSamples for a concrete u0 has no effect") GPODESampledEnsembleProblem(
            sprob, NoReduction(); nGPSamples=ntraj, nInValSamples=10
        )

        @test ensprob isa GPODESampledEnsembleProblem
        @test ensprob.prob isa EnsembleProblem

        enssol = solve(ensprob, Euler(); dt=h)

        @test enssol isa EnsembleSolution
        @test length(enssol) == ntraj

        # can analyze with 
        using DifferentialEquations.EnsembleAnalysis
        ts_stats = timeseries_steps_meanvar(enssol)

        ### testing online sampling
        nbtch = 100

        ## test manual OnlineStat (Normal)
        oensprob1 = GPODESampledEnsembleProblem(sprob; nGPSamples=ntraj)
        oensprob1_ref = GPODESampledEnsembleProblem(
            sprob, OnlineReduction(FitGenNormal()); nGPSamples=ntraj
        )

        @test oensprob1 isa GPODESampledEnsembleProblem{<:OnlineReduction}
        @test typeof(oensprob1) == typeof(oensprob1_ref)

        @test_throws ArgumentError GPODESampledEnsembleProblem(
            sprob; nGPSamples=ntraj, prob_func=x -> 2 * x
        )
        @test_throws ArgumentError GPODESampledEnsembleProblem(
            sprob; nGPSamples=ntraj, reduction=x -> 2 * x
        )
        # test error if not stepsize or saveat given
        @test_throws ArgumentError solve(oensprob1, Tsit5())
        # test for error for trajectory keyword
        @test_throws ArgumentError solve(
            oensprob1, Euler(); dt=h, trajectories=ntraj, batch_size=nbtch
        )

        oenssol1 = solve(oensprob1, Euler(); dt=h, batch_size=nbtch)

        @test oenssol1 isa ODESolution
        @test all(nobs.(oenssol1.u) .== ntraj)
        @test value.(oenssol1.u) isa Vector{Tuple{Float64,Float64}}

        ## Only Mean

        oensprob2 = GPODESampledEnsembleProblem(
            sprob, OnlineReduction(Mean()); nGPSamples=ntraj
        )

        oenssol2 = solve(oensprob2, Euler(); dt=h, batch_size=nbtch)

        @test all(nobs.(oenssol2.u) .== ntraj)
        @test value.(oenssol2.u) isa Vector{Float64}

        @test mean.(oenssol2.u) ≈ ts_stats[1].u atol = 1e-1
    end
    #####################################################
    @testset "Initial Value Distribution" begin
        u0 = Normal(1.0, 0.1)

        sprob = SampledGPODEProblem(gpf, xrange, u0, tspan)

        # testing single sample
        @test sprob isa SampledGPODEProblem
        @test sprob.prob.p isa
            GPDiffEq.PullSolversModule.Interpolations.AbstractInterpolationWrapper

        ssol = solve(sprob, Euler(); dt=h)

        @test ssol isa ODESolution

        # "normal" ensemble

        ntraj = 500
        ninval = 50
        # test missing nInValSamples
        @test_throws ArgumentError GPODESampledEnsembleProblem(
            sprob, NoReduction(); nGPSamples=ntraj
        )

        ensprob = GPODESampledEnsembleProblem(
            sprob, NoReduction(); nGPSamples=ntraj, nInValSamples=ninval
        )
        @test ensprob isa GPODESampledEnsembleProblem
        @test ensprob.prob isa EnsembleProblem

        enssol = solve(ensprob, Euler(); dt=h)

        @test enssol isa EnsembleSolution
        @test length(enssol) == ntraj * ninval

        # can analyze with 
        using DifferentialEquations.EnsembleAnalysis
        ts_stats = timeseries_steps_meanvar(enssol)

        ### testing online sampling
        nbtch = 100

        ## test manual OnlineStat (Normal)
        oensprob1 = GPODESampledEnsembleProblem(
            sprob; nGPSamples=ntraj, nInValSamples=ninval
        )
        oensprob1_ref = GPODESampledEnsembleProblem(
            sprob, OnlineReduction(FitGenNormal()); nGPSamples=ntraj, nInValSamples=ninval
        )

        @test oensprob1 isa GPODESampledEnsembleProblem{<:OnlineReduction}
        @test typeof(oensprob1) == typeof(oensprob1_ref)

        @test_throws ArgumentError GPODESampledEnsembleProblem(
            sprob; nGPSamples=ntraj, prob_func=x -> 2 * x
        )
        @test_throws ArgumentError GPODESampledEnsembleProblem(
            sprob; nGPSamples=ntraj, reduction=x -> 2 * x
        )
        # test error if not stepsize or saveat given
        @test_throws ArgumentError solve(oensprob1, Tsit5())

        oenssol1 = solve(oensprob1, Euler(); dt=h, batch_size=nbtch)

        @test oenssol1 isa ODESolution
        @test all(nobs.(oenssol1.u) .== ntraj * ninval)
        @test value.(oenssol1.u) isa Vector{Tuple{Float64,Float64}}

        ## Only Mean

        oensprob2 = GPODESampledEnsembleProblem(
            sprob, OnlineReduction(Mean()); nGPSamples=ntraj, nInValSamples=ninval
        )

        oenssol2 = solve(oensprob2, Euler(); dt=h, batch_size=nbtch)

        @test all(nobs.(oenssol2.u) .== ntraj * ninval)
        @test value.(oenssol2.u) isa Vector{Float64}

        @test mean.(oenssol2.u) ≈ ts_stats[1].u atol = 1e-1
    end
end

@testset "2D" begin
    # Setup: 

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

    # ### A GPODE problem

    # We define a `GPODEProblem` with the GP as the vector field.
    h = 0.002
    tspan = (0.0, 2.0)
    gpff = GPODEFunction(fp)
    x2range = [xrange, xrange]

    @testset "Concrete initial value" begin
        u0 = [2.0; 0.0]
        # my gp solver
        prob = GPODEProblem(gpff, u0, tspan)

        gpsol = solve(prob, PULLEuler(); dt=h)

        # single sample
        sprob = SampledGPODEProblem(gpff, x2range, u0, tspan)

        ssol = solve(sprob, Euler(); dt=h)

        # "normal" ensemble
        ntraj = 500
        ensprob = GPODESampledEnsembleProblem(sprob, NoReduction(); nGPSamples=ntraj)

        enssol = solve(ensprob, Euler(); dt=h)

        using DifferentialEquations.EnsembleAnalysis

        sm = timeseries_steps_mean(enssol).u
        sv = [cov(getindex.(enssol.u, i)) for i in 1:length(enssol.u[1])]

        ### MvNormal
        nbtch = 100

        oensprob1 = GPODESampledEnsembleProblem(sprob; nGPSamples=ntraj)
        oenssol1 = solve(oensprob1, Euler(); dt=h, batch_size=nbtch)

        ## correct form
        @test oenssol1 isa ODESolution
        @test all(nobs.(oenssol1.u) .== ntraj)
        @test mean.(oenssol1.u) isa Vector{Vector{Float64}}
        @test cov.(oenssol1.u) isa Vector{Matrix{Float64}}

        ## correct values
        # might need to increase number of samples, atol is bad, but later
        @test all(isapprox.(mean.(oenssol1.u), sm; atol=0.5))
        @test all(isapprox.(cov.(oenssol1.u), sv; atol=0.5))
    end

    @testset "Initial distribution" begin
        u0 = MvNormal([2.0; 0.0], diagm([0.001, 0.001]))
        # my gp solver
        prob = GPODEProblem(gpff, u0, tspan)

        gpsol = solve(prob, PULLEuler(); dt=h)

        # single sample
        sprob = SampledGPODEProblem(gpff, x2range, u0, tspan)

        ssol = solve(sprob, Euler(); dt=h)

        # "normal" ensemble
        ntraj = 100
        ninval = 10
        ensprob = GPODESampledEnsembleProblem(
            sprob, NoReduction(); nGPSamples=ntraj, nInValSamples=ninval
        )

        enssol = solve(ensprob, Euler(); dt=h)

        using DifferentialEquations.EnsembleAnalysis

        sm = timeseries_steps_mean(enssol).u
        sv = [cov(getindex.(enssol.u, i)) for i in eachindex(enssol.u[1])]

        ### MvNormal
        nbtch = 100

        oensprob1 = GPODESampledEnsembleProblem(
            sprob; nGPSamples=ntraj, nInValSamples=ninval
        )
        oenssol1 = solve(oensprob1, Euler(); dt=h, batch_size=nbtch)

        ## correct form
        @test oenssol1 isa ODESolution
        @test all(nobs.(oenssol1.u) .== ntraj * ninval)
        @test mean.(oenssol1.u) isa Vector{Vector{Float64}}
        @test cov.(oenssol1.u) isa Vector{Matrix{Float64}}

        ## correct values
        # might need to increase number of samples, atol is bad, but later
        @test all(isapprox.(mean.(oenssol1.u), sm; atol=0.5))
        @test all(isapprox.(cov.(oenssol1.u), sv; atol=0.5))
    end
end