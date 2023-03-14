module DerivativeGPModule

using DocStringExtensions
using Reexport

using AbstractGPs
using LinearAlgebra
using Statistics
using StatsBase
using Zygote

export DerivativeGP, DerivativeKernelCollection
export differentiate

include("derivkernels.jl")
include("derivgp.jl")
include("derivposterior.jl")

end