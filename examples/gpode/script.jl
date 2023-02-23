# # Trajectory Uncertainty

# When the vector field $\dot{x} = f(x)$ is a Gaussian process, we have uncertainty in the vector field, which we can lift to the trajectory. In this example we show how to use the approximate `PULL` solvers, introduced in [this paper](https://arxiv.org/abs/2211.11103), to propagate this uncertainty.

#md # !!! note "Note"
#md #     The implementation of the `PULL` solvers is still in progress.

# ## 1D Example
# ### Setup
using GPDiffEq
using Plots
using LinearAlgebra
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
u0 = [1.0]
tspan = (0.0, 8.0)
ff = GPODEFunction(fp)

prob = GPODEProblem(ff, u0, tspan)

# and integrate with the PULL Euler solver. 
sol = solve(prob, PULLEuler(); dt=0.1)

### Old method for comparison
u0_old = 1.0
old_prob = GPODEProblem(ff, u0_old, tspan)

sol_old = GPDiffEq.PullSolversModule.old_solve(old_prob, PULLEuler(); dt=0.1)

# plot(sol; label="PULL Euler", legend=:bottomright)
p = plot(; layout=(2, 1))
plot!(
    p,
    sol.t,
    only.(getfield.(sol.u, :μ));
    ribbons=sqrt.(only.(getfield.(sol.u, :Σ))),
    subplot=1,
)
plot!(sol_old.t, getfield.(sol_old.u, :val); ribbons=getfield.(sol_old.u, :err), subplot=1)
plot!(p, sol.t, sqrt.(only.(getfield.(sol.u, :Σ))); subplot=2, label="new")
plot!(p, sol_old.t, getfield.(sol_old.u, :err); subplot=2, label="old", linestyle=:dash)
# ## 2D Example

# ### Setup

function fun(x)
    return [-0.1 2.0; -2.0 -0.1] * (x .^ 3)
end
xrange = range(-2.2, 2.2; length=6)
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
gpff = GPODEFunction(fp)

# Plot the GP

ug = range(-2.0, 2.0; length=6)
ug = vcat.(ug, ug')[:]
gp_pred_mean = gpff.(ug)
sf = maximum(norm.(gp_pred_mean))
dug = fun.(ug)
quiver(
    getindex.(ug, 1),
    getindex.(ug, 2);
    quiver=(getindex.(dug, 1) / sf, getindex.(dug, 2) / sf),
)
quiver!(
    getindex.(ug, 1),
    getindex.(ug, 2);
    quiver=(getindex.(gp_pred_mean, 1) / sf, getindex.(gp_pred_mean, 2) / sf),
)
# on plotting: should try both density and ellipses

prob = GPODEProblem(gpff, u0, tspan)

# standard Euler
sol = solve(prob, Euler(); dt=0.05)

# and integrate with the PULL Euler solver. 
sol = solve(prob, PULLEuler(); dt=0.05)

h = 0.05
nsteps2 = ceil(Int, (prob.tspan[end] - prob.tspan[1]) / h)
xtest2 = [zeros(2) for _ in 1:(nsteps2 + 1)]
xtest2[1] = prob.u0
for i in 1:nsteps2
    xtest2[i + 1] = xtest2[i] + h * GPDiffEq.PullSolversModule._mean(gp, xtest2[i])
    # xtest2[i + 1] = xtest2[i] + h * gpff(xtest2[i])
end

# very odd, results are different, need to investigate gpff should be the same as _mean???

# plot(sol; label="PULL Euler", legend=:bottomright)
h = 0.05
gp = prob.f.gp
dgp = prob.f.dgp

xtest, a = GPDiffEq.PullSolversModule._init([2.0, 0.0], 10)
n = 1
########

xnval = xtest[n].μ
xnerr = xtest[n].Σ
a[n] = GPDiffEq.PullSolversModule._mean(dgp, xnval)

m = xnval + h * GPDiffEq.PullSolversModule._mean(gp, xnval)

lhist = 150
ldx = max(1, n - lhist)
μo = getfield.(xtest[ldx:n], :μ)

Kr = GPDiffEq.PullSolversModule._cov(gp, reverse(μo), [μo[end]])
ah = ([I] .+ a[(ldx + 1):n] .* h)
ahe = reverse(cumprod(reverse(ah)))

ahr = reduce(hcat, ahe; init=0.5 * I)

cv1 = ahr * Kr
# really wonder if this is correct, and if so, if I can just make it better with A + A'?
cv2 = Kr' * ahr'

# (I + a[n] * h) * xnerr * (I + a[n] * h)'
quadf2((I + a[n] * h), xnerr)
h^2 * (cv1 + cv2)
v = quadf2((I + a[n] * h), xnerr) + h^2 * (cv1 + cv2)
xtest[n + 1] = AbstractGPs.MvNormal(m, v)
n += 1

# plot trajectory: goes off into x= -3 range, unstable? should check stability of learned function and original function
# maybe with grid and determinant of jacobian contour plot?
# initially, eigenvales increase nicely:
# getfield.(eigen.(getfield.(xtest[1:n], :Σ)), :values)
# from my intuition, this should not be possible. The method should be "stable"? But it is explicit Euler, so maybe it isn't?
# proof? 
# with regular Euler, I get a "normal" trajectory? Need to run only the mean part in a loop (renamed variables and all) and compare
# prove that method gets positive variance (by induction, starting with v1 as function a1, a0, ...)

#
# the former is faster, the latter is safer. Could feasibly just run 0.5*(A+A'), but seems a little riskier?
# xnerr is sure to be symmetric, as it comes from the MvNormal, which has already enforced that. 
function quadf2(ahn, Σ)
    C = ahn * Σ * ahn'
    return 0.5 * (C + C')
end
quadf(ahn, Σ) = ahn * Σ * ahn'
function quadfchl(ahn, Σ)
    C = ahn * cholesky(Σ).L
    return C * C'
end
#
########
xnval = getfield.(xtest[n], :val)
xnerr = getfield.(xtest[n], :err)

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

# Can I rewrite, take the whole row, init the reduction with 1/2I which takes care of K(i,i) + K(i,i)?

#
#
#
## just for testing
gp1 = GP(ker)
fx1 = gp1(X, σ_n^2)
fp1 = posterior(fx1, rand(10))