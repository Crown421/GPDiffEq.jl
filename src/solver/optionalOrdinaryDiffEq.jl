#ToDo: only if OrdinaryDiffEq is loaded, need to check if Requires.jl works when loaded via DifferentialEquations.jl
import OrdinaryDiffEq: OrdinaryDiffEqAlgorithm
function SciMLBase.__solve(
    prob::AbstractGPODEProblem, alg::OrdinaryDiffEqAlgorithm; kwargs...
)
    return error("Using ODEAlgorithms with GPODEs not yet implemented, coming soon")
end