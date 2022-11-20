module GPDiffEq

using Reexport
using DocStringExtensions

@reexport using ApproximateGPs
using DiffEqBase
using Flux
using Optimization
using OptimizationOptimJL
using InducingPoints
using Statistics
using StatsBase
using LinearAlgebra
using KernelFunctions
using Zygote

export build_deriv_model
export GPODE
export gp_train

include("solver/PullSolversModule.jl")

@reexport using .PullSoversModule

include("utils.jl")
include("gp_de.jl")
include("train.jl")
include("derivativeGPs/derivkernels.jl")
include("derivativeGPs/derivgp.jl")
include("derivativeGPs/derivposterior.jl")

end # module
