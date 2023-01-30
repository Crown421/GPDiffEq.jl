module PullSolversModule

using DocStringExtensions
using Reexport
# @reexport using DiffEqBase
@reexport using SciMLBase
@reexport using Measurements

using LinearAlgebra
using AbstractGPs
import SciMLBase: AbstractODEAlgorithm
# using OrdinaryDiffEq
# import OrdinaryDiffEq:
#     OrdinaryDiffEqAlgorithm,
#     OrdinaryDiffEqMutableCache,
#     OrdinaryDiffEqConstantCache,
#     alg_order,
#     alg_cache,
#     initialize!,
#     perform_step!,
#     trivial_limiter!,
#     constvalue,
#     @muladd,
#     @unpack,
#     @cache,
#     @..

include("GPDEFunction.jl")
include("GPDEProblem.jl")
# Solvers
## ToDo: Probably don't want OrdinaryDiffEqAlgorithm
# in fact, need some `Abstract` thing here. 
abstract type AbstractPULLAlg <: AbstractODEAlgorithm end

include("PULLEuler.jl")
include("utils.jl")

using Requires
function __init__()
    @require OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed" include(
        "optionalOrdinaryDiffEq.jl"
    )
end

end