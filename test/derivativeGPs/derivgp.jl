@testset "base gp" begin

    # Ensure that GP implements the AbstractGP API consistently.
    @testset "DerivativeGP" begin
        rng, N, N′ = MersenneTwister(123456), 5, 6
        m, k = ZeroMean(), GaussianKernel()
        f = DerivativeGP(k)
        f2 = DerivativeGP(m, k)
        @test f === f2

        x = collect(range(-1.0, 1.0; length=N))
        x′ = collect(range(-1.0, 1.0; length=N′))

        @test mean(f, x) == AbstractGPs._map_meanfunction(m, x)
        dk = GPDiffEq.BothComponentDerivativeKernel(k)
        @test cov(f, x) == kernelmatrix(dk, x)
        AbstractGPs.TestUtils.test_internal_abstractgps_interface(rng, f, x, x′)
    end

    # # Check that mean-function specialisations work as expected.
    @testset "sugar" begin
        @test DerivativeGP(5, Matern32Kernel()).dmean isa ZeroMean
        @test DerivativeGP(Matern32Kernel()).dmean isa ZeroMean
    end
end