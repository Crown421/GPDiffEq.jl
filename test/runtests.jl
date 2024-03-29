using Test

using AbstractGPs
using AbstractGPs: AbstractGP, MeanFunction, ConstMean, ZeroMean, ConstMean, CustomMean
# import AbstractGPs.TestUtils
using GPDiffEq
using KernelFunctions
# import KernelFunctions.TestUtils

using Random

@testset "DerivativeGPs" begin
    include("derivativeGPs/derivkernels.jl")
    include("derivativeGPs/derivgp.jl")
    include("derivativeGPs/derivposterior.jl")
end
@testset "Sampling" begin
    include("solver/sampling.jl")
end
@testset "Solvers" begin
    include("solver/GPDETypes.jl")
    @testset "PULLEuler" begin
        include("solver/PULLEuler.jl")
    end
    include("solver/sample-solver.jl")
    include("solver/utils.jl")
end
@testset "Utils" begin
    include("utils.jl")
end