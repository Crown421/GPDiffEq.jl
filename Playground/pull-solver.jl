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

dfp = differentiate(fp)

## Sampling vector fields
xt = range((baser .+ s_offset)...; length=30)

cols = Makie.Colors.distinguishable_colors(
    4, [Makie.Colors.RGB(1, 1, 1), Makie.Colors.RGB(0, 0, 0)]; dropseed=true
)

x0 = 3.0
tspan = (0.0, 4.0)
tstep = 0.1
tsol = collect(range(tspan...; step=tstep))

mea = mean(fp, ts)
st = sqrt.(var(fp, ts))

begin
    maxsteps = 30
    # setup solver
    nsteps = ceil(Int, (tspan[end] - tspan[1]) / tstep)
    xe = zeros(Measurement{Float64}, nsteps + 1)
    av = zeros(nsteps + 1)
    xe[1] = x0

    # plot setup
    fig = Figure(; resolution=(1000, 700))
    display(fig)
    p = fig[1, 1] = GridLayout()
    p1 = Axis(p[1, 1]; xlabel="x", ylabel="f(x)", title="Gaussian Process ODE")
    p2 = Axis(p[1, 2]; xlabel="t", ylabel="x", title="Trajectories")

    # plot overall GP
    lines!(p1, ts, mea; color=cols[4], label="GP", linewidth=3)
    band!(p1, ts, mea .- st, mea .+ st; color=(cols[4], 0.4))

    # plot initial value
    scatter!(
        p1, [x0], mean(fp, [x0]); color=cols[2], markersize=18, label="x_0", marker=:xcross
    )
    scatter!(p2, [0], [x0]; color=cols[2], markersize=18, label="x_0", marker=:xcross)

    # start steps
    xlims!(p1, (1.05, 3.55))
    ylims!(p1, (-3.5, 0.4))
    xlims!(p2, (-0.1, (maxsteps + 1) * tstep))
    ylims!(p2, (1.3, 3.05))

    for i in 1:maxsteps
        # i = 2
        GPDiffEq.PullSoversModule.linearized_eulerstep!(
            fp, dfp, xe, av, tstep, i; lhist=150
        )

        a = av[end]
        m = xe[i].val
        s = xe[i].err
        mn = xe[i + 1].val
        sn = xe[i + 1].err

        vb = sqrt(var(fp, [m])[1])

        meat = mean(fp, [m, mn])

        band!(p1, [m, mn], meat .- vb, meat .+ vb; color=(cols[3], 0.4))
        lines!(p1, [m, mn], meat; color=cols[3], linestyle=:dash, linewidth=3)

        # xlims!(
        #     p1,
        #     (
        #         minimum(getfield.(xe[1:(i + 1)], :val)),
        #         maximum(getfield.(xe[1:(i + 1)], :val)),
        #     ) .* (0.98, 1.02),
        # )
        # ylims later

        scatter!(p1, [mn], [meat[2]]; color=cols[2], markersize=10)

        ofc = 0.00
        lines!(
            p2,
            [(i - 1) * tstep + ofc, i * tstep],
            [m, mn];
            color=cols[4],
            linewidth=3,
            depth_shift=1.0f0,
        )
        band!(
            p2,
            [(i - 1) * tstep + ofc, i * tstep],
            [m - s, mn - sn],
            [m + s, mn + sn];
            color=(cols[4], 0.4),
            depth_shift=0.1f0,
        )

        scatter!(p2, [i * tstep], [mn]; color=cols[2], markersize=10, depth_shift=0.0f0)
        lines!(
            p2, [i * tstep, i * tstep], [mn - sn, mn + sn]; color=cols[2], depth_shift=0.0f0
        )

        sleep(0.2)
    end
end

# need to plot approx "tube" (with std)

# plot an error bar for x1