using GPDiffEq
using Random
using Test

Random.seed!(1234)

@testset "Scalar" begin
    # test object creation and structure

    x = rand(3)
    y = rand(3)

    σ_n = 0.001
    gp = GP(SqExponentialKernel())
    fx = gp(x, σ_n^2)
    fp = posterior(fx, y)

    ff = GPODEFunction(fp)

    dgp = differentiate(fp)
    ff2 = GPODEFunction{false}(gp, dgp)

    # test constructor
    # @test ff == ff2

    # test type
    @test ff isa GPDiffEq.PullSolversModule.AbstractGPODEFunction
    @test ff isa SciMLBase.AbstractODEFunction

    # test contents
    @test fp == ff.gp
    @test dgp == ff.dgp == ff2.dgp

    # test GPDEProblem
    prob = GPODEProblem(ff, 1.0, (0.0, 1.0))

    @test prob isa SciMLBase.AbstractODEProblem
    @test prob isa GPDiffEq.PullSolversModule.AbstractGPODEProblem

    @test prob.u0 == 1.0
    @test prob.tspan == (0.0, 1.0)

    # test gpdefun(x)
    @test abs(ff(x[1]) - y[1]) < σ_n * 2
end

@testset "2D" begin
    x = [rand(2) for i in 1:3]
    y = ColVecs(rand(2, 3))

    xMO, yMO = prepare_isotopic_multi_output_data(x, y)

    σ_n = 0.001
    mker = IndependentMOKernel(SqExponentialKernel())

    gp = GP(mker)
    fx = gp(xMO, σ_n^2)
    fp = posterior(fx, yMO)

    ff = GPODEFunction(fp)

    @test sum(abs.(ff(x[1]) - y[1])) < σ_n * 2
end