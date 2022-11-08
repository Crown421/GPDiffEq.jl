module GPDiffEq

using Reexport

@reexport using ApproximateGPs
using DiffEqBase
using Flux
using GalacticOptim
using InducingPoints
using Statistics
using KernelFunctions
using Zygote

export build_deriv_model
export GPODE
export gp_train

include("utils.jl")
include("gp_de.jl")
include("train.jl")
include("derivkernels.jl")
include("derivgp.jl")

end # module
