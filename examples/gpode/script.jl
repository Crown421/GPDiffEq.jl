# # Trajectory Uncertainty

# When the vector field $\dot{x} = f(x)$ is a Gaussian process, we have uncertainty in the vector field, which we can lift to the trajectory. In this example we show how to use the approximate `PULL` solvers, introduced in [this paper](https://arxiv.org/abs/2211.11103), to propagate this uncertainty.

#md # !!! note "Note"
#md #     The implementation of the `PULL` solvers is still in progress.

# ## 1D Example
# ### Setup
using GPDiffEq
using Plots

# ### Defining a simple GP
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

# ### A GPODE problem

# We define a `GPODEProblem` with the GP as the vector field.
u0 = 1.0
tspan = (0.0, 4.0)
ff = GPODEFunction(fp)

prob = GPODEProblem(ff, u0, tspan)

# and integrate with the PULL Euler solver. 
sol = solve(prob, PULLEuler(); dt=0.1)

plot(sol; label="PULL Euler", legend=:bottomright)

# ## 2D Example

# ### Setup

function fun(x)
    return [-0.1 2.0; -2.0 -0.1] * (x .^ 3)
end
xrange = range(-1, 1; length=4)
x = collect.(Iterators.product(xrange, xrange))[:]
y = fun.(x)
## This is annoying UX, needs fix
y = ColVecs(reduce(hcat, y))
xMO, yMO = prepare_isotopic_multi_output_data(x, y)

# ### Defining a Multi-Output GP
σ_n = 3e-2
ker = SqExponentialKernel()
mker = IndependentMOKernel(ker)

gp = GP(mker)
fx = gp(xMO, σ_n)
fp = posterior(fx, yMO)

# ### A GPODE problem

# We define a `GPODEProblem` with the GP as the vector field.
u0 = [2.0; 0.0]
tspan = (0.0, 4.0)
ff = GPODEFunction(fp)

prob = GPODEProblem(ff, u0, tspan)

# and integrate with the PULL Euler solver. 
sol = solve(prob, PULLEuler(); dt=0.1)

# plot(sol; label="PULL Euler", legend=:bottomright)
h = 0.1
gp = prob.f.gp
dgp = prob.f.dgp

xtest, avtest = GPDiffEq.PullSolversModule._init([1.0, 2.0], 10)
n = 1
xnval = getfield.(xtest[n], :val)
xnerr = getfield.(xtest[n], :err)
avtest[n] = GPDiffEq.PullSolversModule._mean(dgp, xnval)

#
m = xnval + h * GPDiffEq.PullSolversModule._mean(gp, xnval)

lhist = 150
ldx = max(1, n - lhist)
μo = map(x -> getfield.(x, :val), xtest[ldx:n])
K = GPDiffEq.PullSolversModule._cov(gp, μo)
d = length(xnval)
Kbr = K[(end - (d - 1)):end, :]
vb = Kbr[(end - (d - 1)):end, (end - (d - 1)):end]
ah = ([I] .+ avtest[(ldx + 1):(n - 1)] .* h)
ahe = reverse(vcat([I], cumprod(reverse(ah))))
# cv = sum(ahe .* Kbr[:, 1:(end - d)])
cv = permutedims(Kbr[:, 1:(end - d)] * reduce(vcat, ahe))

cv = isempty(cv) ? zeros(d, d) : cv

v =
    (I + avtest[n] * h) * diagm(xnerr .^ 2) * (I + avtest[n] * h)' +
    h^2 * vb +
    2 * h^2 * (I + avtest[n] * h) * cv
n += 1

#
#
#
## just for testing
gp1 = GP(ker)
fx1 = gp1(X, σ_n^2)
fp1 = posterior(fx1, rand(10))