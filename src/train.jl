"""
Simplified copy of `sciml_train`

```julia
gp_train(loss, θ, opt=ADAGrad(0.5), adtype=Optimization.AutoZygote(), args...;
lower_bounds=nothing, upper_bounds=nothing, maxiters=1000, kwargs...,)
```

"""
function gp_train(
    loss,
    θ,
    opt=ADAGrad(0.5),
    adtype=Optimization.AutoZygote(),
    args...;
    lower_bounds=nothing,
    upper_bounds=nothing,
    maxiters=1000,
    kwargs...,
)
    optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
    optfunc = Optimization.instantiate_function(optf, θ, adtype, nothing)
    optprob = Optimization.OptimizationProblem(
        optfunc, θ; lb=lower_bounds, ub=upper_bounds, kwargs...
    )

    return Optimization.solve(optprob, opt, args...; maxiters, kwargs...)
end