module GPDiffEq

using ApproximateGPs
using DiffEqBase
using Flux
using GalacticOptim
using InducingPoints
using KernelFunctions
using Zygote

export build_deriv_model
export GPODE
export gp_train

include("utils.jl")
include("gp_de.jl")
include("train.jl")
include("derivkernels.jl")

end # module
