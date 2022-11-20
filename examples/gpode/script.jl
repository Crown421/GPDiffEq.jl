# # Trajectory Uncertainty

# When the vector field $\dot{x} = f(x)$ is given by a Gaussian process, we have uncertainty in the vector field, which we can lift to the trajectory. In this example we show how to use the approximate `PULL` solvers. For more details, see [link].

# ## Setup
using GPDiffEq
using Plots

# ## Defining a simple GP
# In this example we sample the vector field values directly instead of learning them from a trajectory. 

ts = range(-4.25, 4.252; length=100)
f(x) = x * cos(x)

X = range(-3.0, 3.0; length=10)
σ_n = 0.1
y = f.(X) .+ σ_n * randn(length(X))
p = plot(ts, f.(ts); label="truth", xlabel="x", ylabel="f(x)")
scatter!(p, X, y; label="data")

# With this data, we define the GP. 

ker = SqExponentialKernel()
gp = GP(ker)
fx = gp(X, σ_n^2)

fp = posterior(fx, y)

plot!(p, ts, mean(fp, ts); ribbons=sqrt.(var(fp, ts)), label="GP posterior")

# ## A GPODE problem

# We define a `GPODEProblem` with the GP as the vector field.

ff = GPODEFunction(fp)

prob = GPODEProblem(ff, 1.0, (ts[1], ts[end]))