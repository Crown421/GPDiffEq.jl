module GPDiffEq

using Reexport
using DocStringExtensions

@reexport using ApproximateGPs
using DiffEqBase
using Flux
using Optimization
using OptimizationOptimJL
# coming soon
# using InducingPoints
using Statistics
using StatsBase
using LinearAlgebra
using KernelFunctions

export build_deriv_model
export GPODE
export gp_train

include("derivativeGPs/DerivativeGPModule.jl")
include("solver/PullSolversModule.jl")

@reexport using .PullSolversModule
@reexport using .DerivativeGPModule

include("utils.jl")
include("train.jl")

end # module
