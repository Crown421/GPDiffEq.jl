using FiniteDiff
using GPDiffEq

function test_derivative(kernel::Kernel)
    name = string(typeof(kernel).name.wrapper)
    @testset "$name" begin
        dkc = DerivativeKernelCollection(kernel)
        x1 = 1.0
        x2 = 1.1

        # Finite Differences for comparison
        find10(t1, t2) = FiniteDiff.finite_difference_derivative(t1 -> kernel(t1, t2), t1)
        find01(t1, t2) = FiniteDiff.finite_difference_derivative(t2 -> kernel(t1, t2), t2)
        function find11(t1, t2)
            return FiniteDiff.finite_difference_derivative(
                t2 -> FiniteDiff.finite_difference_derivative(t1 -> kernel(t1, t2), t1), t2
            )
        end

        @test dkc.d10(x1, x2) ≈ find10(x1, x2) atol = 1e-6
        @test dkc.d01(x1, x2) ≈ find01(x1, x2) atol = 1e-6
        @test dkc.d11(x1, x2) ≈ find11(x1, x2) atol = 1e-6
    end
end

@testset "derivative kernel" begin
    @test GPDiffEq.FirstComponentDerivativeKernel <: Kernel
    @test GPDiffEq.SecondComponentDerivativeKernel <: Kernel
    @test GPDiffEq.BothComponentDerivativeKernel <: Kernel

    bker = GaussianKernel()

    dkc = DerivativeKernelCollection(bker)

    ### Type checks
    @test dkc.d10 isa GPDiffEq.FirstComponentDerivativeKernel
    @test dkc.d01 isa GPDiffEq.SecondComponentDerivativeKernel
    @test dkc.d11 isa GPDiffEq.BothComponentDerivativeKernel

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