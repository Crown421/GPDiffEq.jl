using Test

using AbstractGPs
using AbstractGPs: AbstractGP, MeanFunction, ConstMean, ZeroMean, ConstMean, CustomMean
# import AbstractGPs.TestUtils
using GPDiffEq
using KernelFunctions
# import KernelFunctions.TestUtils

using Random

include("derivativeGPs/derivkernels.jl")
include("derivativeGPs/derivgp.jl")
include("derivativeGPs/derivposterior.jl")
@testset "Sampling" begin
    include("sampling.jl")
end
@testset "Utils" begin
    include("utils.jl")
end