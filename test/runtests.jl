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