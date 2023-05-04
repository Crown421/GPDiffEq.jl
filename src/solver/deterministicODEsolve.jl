import OrdinaryDiffEq: OrdinaryDiffEqAlgorithm

basic_tgrad(u, p, t) = zero(u)

function SciMLBase.__solve(
    gpprob::AbstractGPODEProblem, alg::OrdinaryDiffEqAlgorithm; kwargs...
)
    # dudt_(u, p, t) = _mean(gpprob.f.gp, u)
    dudt_(u, p, t) = gpprob.f(u)

    ff = ODEFunction{false}(dudt_; tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff, gpprob.u0, gpprob.tspan)
    return solve(prob, alg; kwargs...)

    # return error("Using ODEAlgorithms with GPODEs not yet implemented, coming soon")
end