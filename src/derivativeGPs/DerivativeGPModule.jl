module DerivativeGPModule

using DocStringExtensions
using Reexport

using AbstractGPs
using LinearAlgebra
using Enzyme
using Statistics
using StatsBase

export DerivativeGP, DerivativeKernelCollection
export differentiate

include("derivkernels.jl")
include("derivgp.jl")
include("derivposterior.jl")

end