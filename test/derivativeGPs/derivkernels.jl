@testset "derivative kernel" begin
    bker = GaussianKernel()

    dkc = DerivativeKernelCollection(bker)

    ### Standardised tests.
    # only d11 is a "real" (symmetric) kernel
    KernelFunctions.TestUtils.test_interface(dkc.d11, Vector{Float64})
    # no AD tests yet, but probably be useful for learning
end