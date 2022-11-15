
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
datasize = 10
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
scatter!(p, datatsteps, ode_data[1, :]; markersize=4)#, markerstyle = :star)
scatter!(p, datatsteps, ode_data[2, :]; m=(4, :pentagon), lab="pentagon")#, markerstyle = :star)

# ## Gradient data
# For this example we get gradient observations from our trajectory data via finite differences

# First, we set all necessary variables
scaker = with_lengthscale(SqExponentialKernel(), 1.0)
moker = IndependentMOKernel(scaker)
##ToDo: make ODE data into col_vecs and add number programmatically
x = MOInput(datatsteps, 2)
σ_n = 1e-6
y = ode_data'[:]
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
plot(sol(t_plot); label=["ode" ""], color=[:skyblue :navy], linewidth=3.5)
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
du_pred_mean = reshape(du_pred_mean, :, 2)

du = trueODEfunc.(eachcol(ode_data), 0, 0)
sf = maximum(norm.(du))
quiver(
    ode_data[1, :], ode_data[2, :]; quiver=(getindex.(du, 1) / sf, getindex.(du, 2) / sf)
)
quiver!(
    ode_data[1, :],
    ode_data[2, :];
    quiver=(du_pred_mean[:, 1] / sf, du_pred_mean[:, 2] / sf),
)

# This leaves us with `u` and `udot` pairs as in the input and output:
u = ColVecs(ode_data)
udot = ColVecs(du_pred_mean')

# ## Building a model
# Now we build a model for the the ODE. 

scaker = with_lengthscale(SqExponentialKernel(), ones(2))
moker = IndependentMOKernel(scaker)

u_mo = MOInput(u, 2)
σ_n = 1e-6
y = reduce(vcat, udot.X)
nothing #hide

# and build a posterior GP
gpmodel = GP(moker)
fin_gpmodel = gpmodel(u_mo, σ_n)
post_gpmodel = posterior(fin_gpmodel, y)

# and optimize
loss, buildgppost = gp_negloglikelihood(fin_gpmodel, u_mo, y)

p0 = log.(ones(2))
unfl(x) = exp.(x)

optf = Optimization.OptimizationFunction((x, p) -> (loss ∘ unfl)(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p0)

optp = Optimization.solve(optprob, NelderMead(); maxiters=300)

optparams = unfl(optp)

# We build a posterior GP with the optimized parameters,
optpost = buildgppost(optparams)
nothing #hide

# and incorporate into a GP ode model. Unfortunately, this does not currently match the previous implementation. 

gpode = GPODE(optpost, tspan)
gpsol = gpode(u0)

plot(gpsol)
