# using AbstractGPs
using GPDiffEq
using Random
using Test

Random.seed!(1234)
# _cov

@testset "Cov" begin

    # 1D (scalar)
    @testset "Scalar" begin
        x = rand(3)
        y = rand(3)

        gp = GP(SqExponentialKernel())
        fx = gp(x, 0.001^2)
        fp = posterior(fx, y)

        xin = rand(2)

        c1 = GPDiffEq.PullSolversModule._cov(fp, xin)
        c2 = cov(fp, xin)

        @test c1 ≈ c2
    end

    @testset "2D" begin
        x = [rand(2) for i in 1:3]
        y = ColVecs(rand(2, 3))

        xMO, yMO = prepare_isotopic_multi_output_data(x, y)

        mker = IndependentMOKernel(SqExponentialKernel())
        gp = GP(mker)
        fx = gp(xMO, 0.001^2)
        fp = posterior(fx, yMO)

        xin = [rand(2) for _ in 1:2]
        c1 = GPDiffEq.PullSolversModule._cov(fp, xin)

        xinMO = KernelFunctions.MOInputIsotopicByFeatures(xin, 2)

        c2 = cov(fp, xinMO)

        @test c1 ≈ c2
    end
end

@testset "_makeMOInput" begin
    x = rand(2)
    xMOF = KernelFunctions.MOInputIsotopicByOutputs(x, 2)

    x2 = rand(2)

    x2MOF = GPDiffEq.PullSolversModule._makeMOInput(x2, xMOF)
    @test x2MOF isa typeof(xMOF)
    @test length(x2MOF) == 4
    @test x2MOF.x == x2

    xMOO = KernelFunctions.MOInputIsotopicByOutputs(x, 2)

    x2MOO = GPDiffEq.PullSolversModule._makeMOInput(x2, xMOO)
    @test x2MOO isa typeof(xMOO)
    @test length(x2MOO) == 4
    @test x2MOO.x == x2
end