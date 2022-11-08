module GPDiffEq

using Reexport

@reexport using ApproximateGPs
using DiffEqBase
using Flux
using GalacticOptim
using InducingPoints
using Statistics
using LinearAlgebra
using KernelFunctions
using Zygote

export build_deriv_model
export GPODE
export gp_train

include("utils.jl")
include("gp_de.jl")
include("train.jl")
include("derivativeGPs/derivkernels.jl")
include("derivativeGPs/derivgp.jl")
include("derivativeGPs/derivposterior.jl")

end # module
