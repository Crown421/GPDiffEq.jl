module GPDiffEq

using KernelFunctions
using Flux
using InducingPoints
using ApproximateGPs
using GalacticOptim
using Zygote
using ProgressMeter

include("utils.jl")
include("gp_de.jl")
include("train.jl")
include("derivkernels.jl")

end # module
