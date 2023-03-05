using GLMakie
using Colors
using LinearAlgebra
using DifferentialEquations
# using Interpolations
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

nsample_v = vcat(ones(Int, 10), 2 * ones(Int, 10), 5 * ones(Int, 10), 15 * ones(Int, 10))
samples = [[sample_function(fp, xt) for _ in 1:n] for n in nsample_v];
nsampl = length(nsample_v)
## Colors
cols = Makie.Colors.distinguishable_colors(
    4, [Makie.Colors.RGB(1, 1, 1), Makie.Colors.RGB(0, 0, 0)]; dropseed=true
)

s_cols = range(colorant"grey0", colorant"grey80"; length=nsampl)

x0 = 3.0
tspan = (0.0, 4.0)
tstep = 0.1
tsol = collect(range(tspan...; step=tstep))

function solveODE(f_itp, x0, tspan, tstep)
    ff = (u, p, t) -> f_itp(u)
    prob = ODEProblem(ff, x0, tspan)
    sol = solve(prob, Tsit5(); saveat=tstep)
    return sol.u
end

## Plotting
# fig = Figure(; resolution=(1000, 700))
# display(fig)

begin
    fig = Figure(; resolution=(1200, 700))
    display(fig)
    p = fig[1, 1] = GridLayout()
    p1 = Axis(p[1, 1]; xlabel="x", ylabel="f(x)", title="Gaussian Process ODE")
    p2 = Axis(p[1, 2]; xlabel="t", ylabel="x", title="Trajectories")

    # Opacity stuff
    # opacity_v = reverse(range(0.01, 0.99; length=nsampl))
    opacity_v = reverse(cumsum(nsample_v) / sum(nsample_v))
    opacity_index = Observable(1)

    sampl_color_obs = lift(i -> (s_cols[1], opacity_v[i]), opacity_index)
    mean_color_obs = lift(i -> (cols[4], opacity_v[nsampl + 1 - i]), opacity_index)

    stdstep = 0.2
    stdbands = stdstep:stdstep:2.4
    op_factor = 0.8 / length(stdbands)
    band_color_obs = lift(
        i -> (cols[4], op_factor * opacity_v[nsampl + 1 - i]), opacity_index
    )

    ## Plot 1
    # lines!(p1, ts, f.(ts); color=:black, linewidth=2.5, label="true model")
    mea = mean(fp, ts)
    st = sqrt.(var(fp, ts))

    lines!(p1, ts, mea; color=mean_color_obs, label="GP", linewidth=3)
    for factor in stdbands
        band!(p1, ts, mea .- factor * st, mea .+ factor * st; color=band_color_obs)
    end
    # band!(p1, ts, mea .- 3 * st, mea .+ 3 * st; color=band_color_obs)
    # band!(p1, ts, mea .- 2 * st, mea .+ 2 * st; color=band_color_obs)
    # band!(p1, ts, mea .- st, mea .+ st; color=band_color_obs)
    ylims!(p1, (-3.7, 1.2))

    ## Plot 2
    gpff = GPODEFunction(fp)

    gpprob = GPODEProblem(gpff, x0, tspan)

    # and integrate with the PULL Euler solver. 
    gpsol = solve(gpprob, PULLEuler(); dt=0.1)
    gpsolmea = getfield.(gpsol.u, :val)
    gpsolstd = getfield.(gpsol.u, :err)

    lines!(p2, tsol, gpsolmea; color=mean_color_obs, label="GP", linewidth=3)
    for factor in stdbands
        band!(
            p2,
            tsol,
            gpsolmea .- factor * gpsolstd,
            gpsolmea .+ factor * gpsolstd;
            color=band_color_obs,
        )
    end

    ylims!(p2, (1.15, 3.05))

    # for (i, sample) in enumerate(samples[1:3])
    for (i, sample) in enumerate(samples)
        # framerate = 5
        # record(fig, "fading_animation.mp4", 1:length(samples); framerate=framerate) do i
        sample = samples[i]
        y = reduce(hcat, map(f -> f.(ts), sample))
        series!(p1, ts, y'; solid_color=sampl_color_obs, linewidth=1.5)
        # opacity_obs[] = opacity_v[i]
        opacity_index[] = i

        # ODE
        y = solveODE.(sample, Ref(x0), Ref(tspan), Ref(tstep))
        y = reduce(hcat, y)
        series!(p2, tsol, y'; solid_color=sampl_color_obs, linewidth=1.5)
        sleep(0.2)
    end
end

# j = 40
# sample_set = samples[j]

# y = reduce(hcat, solveODE.(sample_set, Ref(x0), Ref(tspan), Ref(tstep)))
# series(tsol, y'; solid_color=:black)