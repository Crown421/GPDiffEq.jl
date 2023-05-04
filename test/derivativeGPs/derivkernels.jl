using FiniteDiff
using GPDiffEq
using Test

function test_derivative(kernel::Kernel)
    name = string(typeof(kernel).name.wrapper)
    @testset "$name" begin
        dkc = DerivativeKernelCollection(kernel)
        @testset "Scalar Input" begin
            x1 = 1.0
            x2 = 1.1

            # Finite Differences for comparison
            function find10(t1, t2)
                return FiniteDiff.finite_difference_derivative(t1 -> kernel(t1, t2), t1)
            end
            function find01(t1, t2)
                return FiniteDiff.finite_difference_derivative(t2 -> kernel(t1, t2), t2)
            end
            function find11(t1, t2)
                return FiniteDiff.finite_difference_derivative(
                    t2 -> FiniteDiff.finite_difference_derivative(t1 -> kernel(t1, t2), t1),
                    t2,
                )
            end

            @test dkc.d10(x1, x2) ≈ find10(x1, x2) atol = 1e-6
            @test dkc.d01(x1, x2) ≈ find01(x1, x2) atol = 1e-6
            @test dkc.d11(x1, x2) ≈ find11(x1, x2) atol = 1e-6
        end
        @testset "Vector Input" begin
            x1 = [1.0, 2.0]
            x2 = [1.1, 2.1]

            # Finite Differences for comparison
            function find10(t1, t2)
                return FiniteDiff.finite_difference_jacobian(t1 -> kernel(t1, t2), t1)[:]
            end
            function find01(t1, t2)
                return FiniteDiff.finite_difference_jacobian(t2 -> kernel(t1, t2), t2)[:]
            end
            function find11(t1, t2)
                return FiniteDiff.finite_difference_hessian(
                    t -> kernel(t[1:2], t[3:4]), vcat(t1, t2)
                )[
                    1:2, 3:4
                ]
            end

            @test dkc.d10(x1, x2) ≈ find10(x1, x2) atol = 1e-6
            @test dkc.d01(x1, x2) ≈ find01(x1, x2) atol = 1e-6
            @test dkc.d11(x1, x2) ≈ find11(x1, x2) atol = 1e-6
        end
        @testset "MO Input" begin
            mker = IndependentMOKernel(kernel)
            mdkc = DerivativeKernelCollection(mker)

            @testset "Scalar" begin
                x1 = 1.0
                x2 = 2.0
                x1MO = (x1, 1)
                x2MO = (x2, 1)

                @test mdkc.d10(x1MO, x2MO) ≈ dkc.d10(x1, x2) atol = 1e-6
                @test mdkc.d01(x1MO, x2MO) ≈ dkc.d01(x1, x2) atol = 1e-6
                @test mdkc.d11(x1MO, x2MO) ≈ dkc.d11(x1, x2) atol = 1e-6
            end
            @testset "Vector" begin
                x1 = [1.0, 2.0]
                x2 = [2.0, 3.0]
                x1MO = (x1, 1)
                x2MO = (x2, 1)

                @test mdkc.d10(x1MO, x2MO) ≈ dkc.d10(x1, x2) atol = 1e-6
                @test mdkc.d01(x1MO, x2MO) ≈ dkc.d01(x1, x2) atol = 1e-6
                @test mdkc.d11(x1MO, x2MO) ≈ dkc.d11(x1, x2) atol = 1e-6
            end
        end
    end
end

@testset "derivative kernel" begin
    @test GPDiffEq.DerivativeGPModule.Derivative10Kernel <: Kernel
    @test GPDiffEq.DerivativeGPModule.Derivative01Kernel <: Kernel
    @test GPDiffEq.DerivativeGPModule.Derivative11Kernel <: Kernel

    bker = GaussianKernel()

    dkc = DerivativeKernelCollection(bker)

    ### Type checks
    @test dkc.d10 isa GPDiffEq.DerivativeGPModule.Derivative10Kernel
    @test dkc.d01 isa GPDiffEq.DerivativeGPModule.Derivative01Kernel
    @test dkc.d11 isa GPDiffEq.DerivativeGPModule.Derivative11Kernel

    @test dkc.d10(1.0, 1.1) isa Real
    @test dkc.d01(1.0, 1.1) isa Real
    @test dkc.d11(1.0, 1.1) isa Real

    ### Utility and consistency checks
    @test dkc.d10(1.0, 1.1) ≈ dkc.d01(1.1, 1.0) atol = 1e-6

    @test dkc(1.0, 1.1) == dkc.d10(1.0, 1.1)

    ### Standardised tests.
    # only d11 is a "real" (symmetric) kernel
    KernelFunctions.TestUtils.test_interface(dkc.d11, Vector{Float64})
    # no AD tests yet, but probably be useful for learning

    ### Correctness
    test_derivative(bker)
    # ToDo: Add more kernel types, MOKernels, broken tests for Matern
end