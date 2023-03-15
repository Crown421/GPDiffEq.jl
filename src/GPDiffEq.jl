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

using Interpolations

export build_deriv_model
export GPODE
export gp_train

include("derivativeGPs/DerivativeGPModule.jl")
include("solver/PullSolversModule.jl")

@reexport using .PullSolversModule
@reexport using .DerivativeGPModule

include("utils.jl")
# include("gp_de.jl")
include("train.jl")
include("sampling.jl")
# include("derivativeGPs/derivkernels.jl")
# include("derivativeGPs/derivgp.jl")
# include("derivativeGPs/derivposterior.jl")

end # module
