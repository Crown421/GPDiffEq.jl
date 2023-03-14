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
        dk = GPDiffEq.DerivativeGPModule.BothComponentDerivativeKernel(k)
        @test cov(f, x) == kernelmatrix(dk, x)
        AbstractGPs.TestUtils.test_internal_abstractgps_interface(rng, f, x, x′)
    end

    # # Check that mean-function specialisations work as expected.
    @testset "sugar" begin
        @test DerivativeGP(5, Matern32Kernel()).dmean isa ZeroMean
        @test DerivativeGP(Matern32Kernel()).dmean isa ZeroMean
    end

    ### Custom mean functions
    @testset "Custom Mean" begin
        rng = MersenneTwister(123456)

        k = GaussianKernel()
        cm = sin #x -> sin(x)
        m = AbstractGPs.CustomMean(cm)

        df = DerivativeGP(m, k)

        dcm = cos

        x = rand(rng, 5)
        @test mean(df, x) ≈ dcm.(x)
    end

    ### Utility
    @testset "Utility" begin
        m, k = ZeroMean(), GaussianKernel()
        f = GP(m, k)
        df = DerivativeGP(m, k)
        df2 = DerivativeGP(f)
        @test df === df2

        @test repr("text/plain", df) ==
            "Derivative of GP f: \n  mean: ZeroMean{Float64}()\n  kernel: Squared Exponential Kernel (metric = Distances.Euclidean(0.0))"

        dfx = df(collect(1:4), 0.1)
        @test repr("text/plain", dfx) ==
            "FiniteGP of Derivative of GP f: \n    mean: ZeroMean{Float64}()\n    kernel: Squared Exponential Kernel (metric = Distances.Euclidean(0.0))\n  x: [1, 2, 3, 4]\n  Σy: [0.1 0.0 0.0 0.0; 0.0 0.1 0.0 0.0; 0.0 0.0 0.1 0.0; 0.0 0.0 0.0 0.1]"
    end
end