using GPDiffEq
using Zygote
using Random
using AbstractGPs: MeanFunction, ConstMean, ZeroMean#, posterior
using LinearAlgebra
using Test

@testset "Posterior Derivative GP" begin
    @testset "DerivativeGP" begin
        rng, N, Ns = MersenneTwister(123456), 5, 6
        σ_n = 1e-4
        m, ker = ZeroMean(), GaussianKernel()
        f = DerivativeGP(ker)

        x = collect(range(-1.0, 1.0; length=N))
        fx = f(x, σ_n)
        y = rand(N)

        xpl = collect(range(-1.0, 1.0; length=20))

        # construct posterior
        fp = posterior(fx, y)

        ## build explicit comparison
        # derivative kernels 
        dker(t1, t2) = first(first(Zygote.gradient(t1 -> ker(t1, t2), t1)))
        function dkerd(t1, t2)
            return first(
                Zygote.gradient(t2 -> first(Zygote.gradient(t1 -> ker(t1, t2), t1)), t2)
            )
        end
        kerd(t1, t2) = first(first(Zygote.gradient(t2 -> ker(t1, t2), t2)))

        K = kernelmatrix(ker, x') + σ_n * I
        Kchol = cholesky(K)

        dgp(xs) = dker.(xs, x)' * (Kchol \ y)
        dvargp(xs::Real) = dkerd.(xs, xs') - first(dker.(xs, x') * (Kchol \ kerd.(x, xs')))
        function dvargp(xs::Array{<:Real})
            return dkerd.(xs, xs') - dker.(xs, x') * (Kchol \ kerd.(x, xs'))
        end
        function dgpcov(xs::Array{<:Real}, ys::Array{<:Real})
            return dkerd.(xs, ys') - dker.(xs, x') * (Kchol \ kerd.(x, ys'))
        end

        @test dgpcov(xpl, xpl) ≈ dvargp(xpl)

        # Verify correctness of derivative posterior mean and variance
        @test mean(fp, xpl) ≈ dgp.(xpl)
        @test var(fp, xpl) ≈ dvargp.(xpl) rtol = 1e-14 atol = 1e-14
        @test cov(fp, xpl) ≈ dvargp(xpl)

        ypl = rand(Ns)

        @test cov(fp, xpl, ypl) ≈ dgpcov(xpl, ypl)

        # Check interface is implemented fully and consistently.
        a = collect(range(-1.0, 1.0; length=N)) * 10
        b = randn(rng, Ns)
        AbstractGPs.TestUtils.test_internal_abstractgps_interface(rng, fp, a, b)
    end

    @testset "Utility" begin
        rng, N, Ns = MersenneTwister(123456), 5, 6
        σ_n = 1e-4
        ker = GaussianKernel()
        f = GP(ker)

        x = range(-1.0, 1.0; length=N)
        fx = f(x, 0.1)
        y = rand(N)

        f_post = posterior(fx, y)

        df_post = differentiate(f_post)

        @test df_post isa AbstractGPs.PosteriorGP{<:DerivativeGP}
        @test df_post.data == f_post.data
        @test df_post.prior isa DerivativeGP
    end
end
