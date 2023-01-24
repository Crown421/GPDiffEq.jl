
# # Spiral ODE

# ## Setup

# Load necessary packages
using ApproximateGPs
using Plots
using LinearAlgebra
using DifferentialEquations
using InducingPoints
using GPDiffEq
using Optimization, OptimizationOptimJL

# First we define an ODE and generate some data points from it. 

u0 = [2.0; 0.0]
datasize = 19
tspan = (0.0, 3.0)
datatspan = (0.0, 1.5)
datatsteps = range(datatspan[1], datatspan[2]; length=datasize)

function trueODEfunc(u, p, t)
    du = similar(u)
    true_A = [-0.1 2.0; -2.0 -0.1]
    return du .= ((u .^ 3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
sol = solve(prob_trueode, Tsit5())
ode_data = Array(sol(datatsteps))

traj = sol(datatsteps);

p = plot(sol)
scatter!(p, datatsteps, ode_data'; markersize=4, color=:black, label=["data" ""])

# ## Gradient data
# For this example we get gradient observations from our trajectory data via finite differences

# First, we set all necessary variables, which includes the Multi-Output Kernel and the inputs and outputs.
# See also the [KernelFunctions.jl Documentation on Multiple Outputs](https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/api/#Inputs-for-Multiple-Outputs).
scaker = with_lengthscale(SqExponentialKernel(), 1.0)
moker = IndependentMOKernel(scaker)

x, y = prepare_isotopic_multi_output_data(collect(datatsteps), ColVecs(ode_data))
σ_n = 1e-3
nothing #hide

# and build a finite GP from them
g = GP(moker)
gt = g(x, σ_n)
gt_post = posterior(gt, y)
nothing #hide

# Now we use the following convenience functions to a loglikelihood loss function and a function to rebuild the gp with the optimal parameters. 
# Note that we use optimize over the logarithm of the parameters, to ensure their positivity. For more details see [this KernelFunctions.jl example](https://juliagaussianprocesses.github.io/KernelFunctions.jl/dev/examples/train-kernel-parameters/)
loss, buildgppost = gp_negloglikelihood(gt, x, y)

p0 = log.([1.0])
unfl(x) = exp.(x)

#optp = gp_train(loss ∘ unfl, p0; show_trace=true, show_every=15)

# Optimizing:
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> (loss ∘ unfl)(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p0)

optp = Optimization.solve(optprob, NelderMead(); maxiters=300)

optparams = unfl(optp)

# Now we can build a FiniteGP with the optimized parameters,
optpost = buildgppost(optparams)
nothing #hide

# which fits pretty well
t_plot = range(datatspan...; length=100)
t_plot_mo = MOInput(t_plot, 2)
opt_pred_mean = mean(optpost, t_plot_mo)
opt_pred_mean = reshape(opt_pred_mean, :, 2)
pred_mean = mean(gt_post, t_plot_mo)
pred_mean = reshape(pred_mean, :, 2)
## pred_cov = diag(cov(optpost, t_plot_mo))
## pred_cov = reshape(pred_cov, :, 2)
## plot!(t_plot, pred_mean; ribbons = pred_cov)
plot(sol; label=["ode" ""], color=[:skyblue :navy], linewidth=3.5)
plot!(
    t_plot,
    pred_mean;
    label=["gp" ""],
    color=[:limegreen :darkgreen],
    linewidth=2.5,
    linestyle=:dashdot,
)
plot!(
    t_plot,
    opt_pred_mean;
    label=["opt. gp" ""],
    color=[:tomato :firebrick],
    linewidth=2.5,
    linestyle=:dash,
)

# GPs are closed under linear operators, which means that we can very easily obtain derivative information:

deriv_post = differentiate(optpost)
du_pred_mean = mean(deriv_post, x)
du_pred_mean = reshape_isotopic_multi_output(du_pred_mean, deriv_post)

du = trueODEfunc.(eachcol(ode_data), 0, 0)
sf = maximum(norm.(du))
quiver(
    ode_data[1, :],
    ode_data[2, :];
    quiver=(getindex.(du, 1) / sf, getindex.(du, 2) / sf),
    label="true",
)
quiver!(
    ode_data[1, :],
    ode_data[2, :];
    quiver=(getindex.(du_pred_mean, 1) / sf, getindex.(du_pred_mean, 2) / sf),
    label="predicted data",
)

# This leaves us with `u` and `udot` pairs as in the input and output:
u = ColVecs(ode_data)
udot = du_pred_mean

# ## Building a model
# Now we build a model for the the ODE. 

scaker = with_lengthscale(SqExponentialKernel(), ones(2))
moker = IndependentMOKernel(scaker)

u_mo, y = prepare_isotopic_multi_output_data(u, udot)
σ_n = 1e-6
nothing #hide

# and build a posterior GP
gpfun = GP(moker)
fin_gpfun = gpfun(u_mo, σ_n)
post_gpfun = posterior(fin_gpfun, y)

# and optimize
loss, buildgppost = gp_negloglikelihood(fin_gpfun, u_mo, y)

p0 = log.(ones(2))
unfl(x) = exp.(x)

optf = Optimization.OptimizationFunction((x, p) -> (loss ∘ unfl)(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p0)

optp = Optimization.solve(optprob, NelderMead(); maxiters=300)

optparams = unfl(optp)

# We build a posterior GP with the optimized parameters,
optpost = buildgppost(optparams)
nothing #hide

# and a GP ODE function

gpff = GPODEFunction(optpost)

# Plotting the vector field
ug = range(-2.0, 2.0; length=6)
ug = vcat.(ug, ug')[:]
gp_pred_mean = gpff.(ug)
quiver!(
    getindex.(ug, 1),
    getindex.(ug, 2);
    quiver=(getindex.(gp_pred_mean, 1) / sf, getindex.(gp_pred_mean, 2) / sf),
    label="GP model",
    legend=:bottomright,
)

# and incorporate into a GP ode model, which can be solved. Initially, the GP ODE corresponds well with the data, but diverges from the true solution as it starts to extrapolate beyond data. 

gpprob = GPODEProblem(gpff, u0, tspan)

gpsol = solve(gpprob, Tsit5())
nothing #hide

# #### Phase plot
plot!(sol; vars=(1, 2), label="ode", linewidth=2, color=:navy)
plot!(gpsol; vars=(1, 2), label="gp", linewidth=2.5, linestyle=:dashdot, color=:darkgreen)
scatter!(ode_data[1, :], ode_data[2, :]; markersize=4, color=:black, label="data")

# # #### Time Series Plots
plot(sol; label=["ode" ""], color=[:skyblue :navy], linewidth=3)
plot!(
    gpsol; label=["gp" ""], color=[:limegreen :darkgreen], linewidth=2, linestyle=:dashdot
)
scatter!(datatsteps, ode_data'; markersize=4, color=:black, label=["data" ""])
