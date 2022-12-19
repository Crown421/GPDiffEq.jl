using GLMakie
using Colors
using LinearAlgebra
using DifferentialEquations
using Interpolations
using GPDiffEq

# key parameters
baser = [-0.2, 4.0]
x0 = 3.0

ylim = (1.3, 3.1)

tspan = (0.0, 4.0)
tstep = 0.1

# generate data
plot_offset = [-0.5, 0.5]
s_offset = [-0.2, 0.2]

ts = range((baser .+ plot_offset)...; length=100)
f(x) = x * cos(x)

X = range(baser...; length=7)
σ_n = 0.2
y = f.(X) .+ σ_n * randn(length(X))
# ; label="data")

ker = SqExponentialKernel()
gp = GP(ker)
fx = gp(X, σ_n^2)

fp = posterior(fx, y)

## Sampling vector fields
xt = range((baser .+ s_offset)...; length=30)

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
nparr = 3
samples = [[exact_sample(fp, xt) for _ in 1:nparr] for _ in 1:15]
push!(samples, samples[1])
# now plotting from interpolation objects
## second thought, want to create interp object separately

### Plotting 
## Setup

cols = Makie.Colors.distinguishable_colors(
    4, [Makie.Colors.RGB(1, 1, 1), Makie.Colors.RGB(0, 0, 0)]; dropseed=true
)

s_cols = range(colorant"grey30", colorant"grey80"; length=nparr + 1)[1:nparr]

begin
    ## figure setup
    fig = Figure(; resolution=(1000, 700))
    display(fig)

    p = fig[1, 1] = GridLayout()
    p1 = Axis(p[1, 1]; xlabel="x", ylabel="f(x)")
    p2 = Axis(p[2, 1]; xlabel="t", ylabel="x")

    ## Initial plot
    lines!(p1, ts, f.(ts); color=:black, linewidth=2.5, label="true model")
    mea = mean(fp, ts)
    st2 = 2 * sqrt.(var(fp, ts))
    lines!(p1, ts, mea; color=cols[4], label="GP")
    band!(p1, ts, mea .- st2, mea .+ st2; color=(cols[4], 0.2))
    # data plotted later for overlay

    # mean GP Solution
    # PULL solver!!
    gpff = GPODEFunction(fp)

    gpprob = GPODEProblem(gpff, x0, (0.0, 4.0))

    # and integrate with the PULL Euler solver. 
    gpsol = solve(gpprob, PULLEuler(); dt=0.1)
    gpsolmea = getfield.(gpsol.u, :val)
    gpsolstd = getfield.(gpsol.u, :err)

    lines!(
        p2, gpsol.t, gpsolmea; color=cols[4], linewidth=2.5, label="trajectory distribution"
    )
    band!(
        p2,
        gpsol.t,
        gpsolmea .- 2 .* gpsolstd,
        gpsolmea .+ 2 .* gpsolstd;
        color=(cols[4], 0.2),
    )

    # wiggly lines:
    # range
    xt_plot = range((baser .+ s_offset)...; length=60)
    tsol = range(tspan...; step=tstep)

    ## Assembling observables
    # helper functions
    function solveODE(f_itp, x0, tspan, tstep)
        ff = (u, p, t) -> f_itp(u)
        prob = ODEProblem(ff, x0, tspan)
        sol = solve(prob, Tsit5(); saveat=tstep)
        return sol.u
    end

    sample_obs = Observable(samples[1])
    # i = 1
    # function plot_for_sample(samples, i)
    # sample_obs = lift(s -> s[i], samples_obs)

    f_itp_obs = lift(s -> create_itp.(Ref(xt), s), sample_obs)
    fs_obs = map(f -> map.(f, Ref(xt_plot)), f_itp_obs)
    fs_obs_r = lift(x -> reduce(hcat, x)', fs_obs)
    # Solution

    # xsol_obs = lift(f -> solveODE(f, x0, tspan, tstep), f_itp_obs)
    xsol_obs = lift(f -> solveODE.(f, Ref(x0), Ref(tspan), Ref(tstep)), f_itp_obs)
    xsol_obs_r = lift(x -> reduce(hcat, x)', xsol_obs)

    ## currently not being plottet
    # dxsol_obs = lift((f, x) -> f.(x), f_itp_obs, xsol_obs)
    # dxsol_obs = lift((f, x) -> map.(f,Ref(x)), f_itp_obs, xsol_obs)
    dx0_obs = lift(f -> map.(f, x0), f_itp_obs)

    ## Initial animated elements state
    series!(p1, xt_plot, fs_obs_r; color=s_cols, linewidth=3.1, label="sampled model")

    # plot data here
    scatter!(p1, X, y; color=cols[1], label="data", markersize=13)

    # line on model
    ## not sure I want them
    # lines!(p1, xsol_obs, dxsol_obs; color=:purple, linewidth=3.8)
    scatter!(p1, [x0], dx0_obs; color=cols[2], markersize=15, label="initial condition")

    # trajectory
    return series!(
        p2,
        collect(tsol),
        xsol_obs_r;
        color=s_cols,
        linewidth=3.3,
        label="sampled model trajectory",
    )

    # some settings
    ylims!(p2, ylim)
end

axislegend(p1; position=:rt)
axislegend(p2; position=:rt)

# plot_for_sample(Ref(samples_obs), 1:nparr)

## Updates/ Animation
# updating all observable

#### Really what happens is that f_itp updates, so that should be an observable, and everything else should be a (series of) lift(s)
for i in 1:(length(samples) - 1)
    for λ in range(0.0, 1.0; length=30)
        # samples_i = samples[i] * (1 - λ) + samples[i + 1] * λ
        sample_i = samples[i] .* (1 - λ) .+ samples[i + 1] .* λ
        sample_obs[] = sample_i

        sleep(0.04)
    end
    println("done $i")
end
# end

### Recording
# ToDo: make twice as fast
begin
    nsecs = length(samples) - 1
    framerate = 30
    accel = 2
    timestamps = range(0, nsecs / accel; step=1 / framerate)
    timestamps = timestamps[1:(end - 1)]

    record(fig, "time_animation.gif", timestamps; framerate=framerate) do t
        i = Int(div(t, 1 / accel)) + 1
        λ = mod(t, 1 / accel) * accel
        sample = samples[i] * (1 - λ) + samples[i + 1] * λ
        sample_obs[] = sample
    end
end
