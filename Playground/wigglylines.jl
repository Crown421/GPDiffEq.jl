using GLMakie
using LinearAlgebra
using DifferentialEquations
using Interpolations
using GPDiffEq

ts = range(-3.5, 3.5; length=100)
f(x) = x * cos(x)

X = range(-3.0, 3.0; length=7)
σ_n = 0.2
y = f.(X) .+ σ_n * randn(length(X))
# ; label="data")

ker = SqExponentialKernel()
gp = GP(ker)
fx = gp(X, σ_n^2)

fp = posterior(fx, y)

## Sampling vector fields
xt = range(-3.2, 3.2; length=30)

# xsample = range(-4.25, 4.252, length = 30)

function exact_sample(pgp, xsample)
    m = mean(pgp, xsample)
    # gp.(xsample)
    ζ = randn(length(xsample))
    # K = Ks(xsample)
    # K = 0.5 * (K+ K') + 1e-12*I
    K = cov(pgp, xsample) + 1e-12 * I
    Kchol = cholesky(K)

    return sample = m .+ Kchol.L * ζ

    # # LinearInterpolation(xsample, sample)
    # itp = interpolate(sample, BSpline(Cubic(Line(OnGrid()))))
    # sitp = Interpolations.scale(itp, xsample)
    # extrapolate(sitp, Line())
    # # extrapolate(sitp, NaN)
end
function create_itp(xsample, sample)
    itp = interpolate(sample, BSpline(Cubic(Line(OnGrid()))))
    sitp = Interpolations.scale(itp, xsample)
    return extrapolate(sitp, Line())
end
samples = [exact_sample(fp, xt) for _ in 1:15]
push!(samples, samples[1])
# now plotting from interpolation objects
## second thought, want to create interp object separately

### Plotting 
## Setup
fig = Figure(; resolution=(1000, 700))
display(fig)

p = fig[1, 1] = GridLayout()
p1 = Axis(p[1, 1]; xlabel="x", ylabel="f(x)")
p2 = Axis(p[2, 1]; xlabel="t", ylabel="x")

## Initial plot
lines!(p1, ts, f.(ts); color=:black)
mea = mean(fp, ts)
st2 = 2 * sqrt.(var(fp, ts))
lines!(p1, ts, mea)
band!(p1, ts, mea .- st2, mea .+ st2; color=(:blue, 0.2))
# ; label="truth", xlabel="x", ylabel="f(x)")
scatter!(p1, X, y; color=:darkred)

# wiggly lines:
# Model
xt_plot = range(-3.2, 3.2; length=60)
sample_obs = Observable(samples[1])

# f_itp = create_itp(xt, samples[1])
f_itp_obs = lift(s -> create_itp(xt, s), sample_obs)
# fxt = f_itp(xt_plot)
fs_obs = lift(f -> f(xt_plot), f_itp_obs)
# Observable(fxt)

# Solution
x0 = 0.45
tspan = (0.0, 4.0)
tstep = 0.1
tsol = range(tspan...; step=tstep)

function solveODE(f_itp, x0, tspan, tstep)
    ff = (u, p, t) -> f_itp(u)
    prob = ODEProblem(ff, x0, tspan)
    sol = solve(prob, Tsit5(); saveat=tstep)
    return sol.u
end
# ff = ODEFunction((u, t, p) -> f_itp(u))
# prob = ODEProblem(ff, x0, (0.0, 4.0))
# sol = solve(prob, Tsit5(); saveat=0.1)

# tsol = sol.t
# xsol = sol.u
# xsol_obs = Observable(xsol)
xsol_obs = lift(f -> solveODE(f, x0, tspan, tstep), f_itp_obs)

# dxsol_obs = lift(f_itp, xsol_obs)
dxsol_obs = lift((f, x) -> f.(x), f_itp_obs, xsol_obs)
# Observable(f_itp(xsol_obs[]))
dx0_obs = lift(f -> [f(x0)], f_itp_obs)
# Observable([f_itp(x0)])

## Initial animated elements state
lines!(p1, xt_plot, fs_obs; color=:grey30, linewidth=3.1)

# line on model
lines!(p1, xsol_obs, dxsol_obs; color=:purple, linewidth=3.8)
scatter!(p1, [x0], dx0_obs; color=:darkred, markersize=15)

# trajectory
lines!(p2, tsol, xsol_obs; color=:purple)

# some settings
ylims!(p2, (-1.9, 1.9))
## Updates/ Animation
# updating all observable

#### Really what happens is that f_itp updates, so that should be an observable, and everything else should be a (series of) lift(s)
for i in 1:(length(samples) - 1)
    for λ in range(0.0, 1.0; length=30)
        sample = samples[i] * (1 - λ) + samples[i + 1] * λ
        sample_obs[] = sample

        # f_itp_sample = create_itp(xt, sample)

        # fxt = f_itp_sample(xt_plot)
        # fs_obs[] = fxt

        # ff = ODEFunction((u, t, p) -> f_itp_sample(u))
        # prob = ODEProblem(ff, x0, (0.0, 4.0))
        # sol = solve(prob, Tsit5(); saveat=0.1)

        # tsol = sol.t
        # xsol = sol.u
        # xsol_obs[] = xsol

        # dxsol_obs[] = f_itp_sample(xsol_obs[])
        # dx0_obs[] = [f_itp_sample(x0)]

        # fs_obs[] = f_sample(xt_plot)
        sleep(0.04)
    end
    println("done $i")
end