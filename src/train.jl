"""
`gp_train`
### Unconstrained Optimization
Simplified copy of `sciml_train`
"""
function gp_train(
    loss,
    θ,
    opt=ADAGrad(0.5),
    adtype=GalacticOptim.AutoZygote(),
    args...;
    lower_bounds=nothing,
    upper_bounds=nothing,
    maxiters=1000,
    kwargs...,
)
    optf = GalacticOptim.OptimizationFunction((x, p) -> loss(x), adtype)
    optfunc = GalacticOptim.instantiate_function(optf, θ, adtype, nothing)
    optprob = GalacticOptim.OptimizationProblem(
        optfunc, θ; lb=lower_bounds, ub=upper_bounds, kwargs...
    )

    return GalacticOptim.solve(optprob, opt, args...; maxiters, kwargs...)
end