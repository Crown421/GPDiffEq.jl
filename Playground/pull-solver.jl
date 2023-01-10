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

    for i in 1:nsteps
        GPDiffEq.PullSoversModule.linearized_eulerstep!(
            fp, dfp, xe, av, tstep, i; lhist=150
        )
    end

    # plot setup
    fig = Figure(; resolution=(1000, 700))
    display(fig)
    p = fig[1, 1] = GridLayout()
    p1 = Axis(p[1, 1]; xlabel="x", ylabel="f(x)", title="Gaussian Process ODE")
    p2 = Axis(p[1, 2]; xlabel="t", ylabel="x", title="Trajectories")

    # plot overall GP
    lines!(p1, ts, mea; color=cols[4], label="GP", linewidth=3.4)
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

    m = getfield.(xe, :val)
    s = getfield.(xe, :err)
    vb = sqrt.(var(fp, m))
    meat = mean(fp, m)

    # begin
    i_obs = Observable(1)

    poly1 = lift(
        i -> vcat(
            Point2.(m[1:i], meat[1:i] .+ vb[1:i]),
            reverse(Point2.(m[1:i], meat[1:i] .- vb[1:i])),
        ),
        i_obs,
    )
    poly!(p1, poly1; color=(cols[3], 0.4))

    m_meat_obs = lift(i -> Point2.(m[1:i], meat[1:i]), i_obs)
    lines!(p1, m_meat_obs; color=cols[3], linestyle=:dash, linewidth=4)

    # xlims!(
    #     p1,
    #     (
    #         minimum(getfield.(xe[1:(i + 1)], :val)),
    #         maximum(getfield.(xe[1:(i + 1)], :val)),
    #     ) .* (0.98, 1.02),
    # )
    # ylims later

    scatter!(p1, m_meat_obs; color=cols[2], markersize=10)

    t_m_obs = lift(i -> Point2.(tstepsf(i), m[1:i]), i_obs)
    lines!(p2, t_m_obs; color=cols[4], linewidth=3)

    poly2 = lift(
        i -> vcat(
            Point2.(tstepsf(i), m[1:i] .+ s[1:i]),
            reverse(Point2.(tstepsf(i), m[1:i] .- s[1:i])),
        ),
        i_obs,
    )
    poly!(p2, poly2; color=(cols[4], 0.4))

    t_m_err_obs = lift(i -> Point3.(tstepsf(i), m[1:i] - s[1:i], m[1:i] + s[1:i]), i_obs)
    rangebars!(p2, t_m_err_obs; color=cols[2], whiskerwidth=6)
    scatter!(p2, t_m_obs; color=cols[2], markersize=10)

    for i in 2:maxsteps
        i_obs[] = i
        sleep(0.2)
    end
end

# ## error plots
# nrpoints = 20
# nrsubsamples = 8

# xs = xe[1:nrpoints]
# ms = getfield.(xs, :val)
# r = collect.(range.(ms[1:(end - 1)], ms[2:end], length=nrsubsamples))
# deleteat!.(r, nrsubsamples)

# # GP mean error
# as = av[1:nrpoints]
# meas = meat[1:nrpoints]
# fr = [(r[i] .- ms[i]) * as[i] .+ meas[i] for i in 1:(nrpoints - 1)]

# series(collect(zip(r, fr)); solid_color=:black)
# gpy = mean(fp, reduce(vcat, r))
# lines!(reduce(vcat, r), gpy; color=:red, linewidth=3)

# err = abs.(gpy .- reduce(vcat, fr))
# lines(reduce(vcat, r), err; color=:red, linewidth=3)

# # GP Var error
# linstd = reduce(vcat, [vb[i] * ones(nrsubsamples - 1) for i in 1:(nrpoints - 1)])
# exstd = sqrt.(var(fp, reduce(vcat, r)))

# lines(reduce(vcat, r), abs.(exstd .- linstd); color=:red, linewidth=3)

# #