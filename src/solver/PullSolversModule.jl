module PullSolversModule

using DocStringExtensions
using Reexport
# @reexport using DiffEqBase
@reexport using SciMLBase
@reexport using Measurements

using AbstractGPs
using LinearAlgebra
using Distributions
import SciMLBase: AbstractODEAlgorithm

using ..DerivativeGPModule

export GPODEFunction
export GPODEProblem

include("GPDETypes.jl")

# Solvers
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