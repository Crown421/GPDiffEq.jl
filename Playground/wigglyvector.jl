using Makie
using LinearAlgebra
using DifferentialEquations
using Interpolations
function trueODEfunc(u)
    du = similar(u)
    true_A = [-0.1 2.0; -2.0 -0.1]
    return du .= ((u .^ 3)'true_A)'
end

# lets make a equidistant grid of points
base = range(-1, 1; length=6)
x = [collect(i) for i in Iterators.product(base, base)][:]

xd = trueODEfunc.(x)
sf = 2 * maximum(norm.(xd))
xdf = xd ./ sf

# plot(rand(4))

###
## Initial plot
arrows(getindex.(x, 1), getindex.(x, 2), getindex.(xdf, 1), getindex.(xdf, 2))

# move the scaling factor out, and sample from "real" field. 
σ = 0.05
sxdfv = [xdf .+ [σ * randn(length(xdf[1])) for _ in 1:length(xdf)] for _ in 1:5]

xd = getindex.(sxdfv[1], 1)
yd = getindex.(sxdfv[1], 2)
xarr = Observable(xd)
yarr = Observable(yd)

itp1 = LinearInterpolation((base, base), reshape(xd, 6, 6))
itp2 = LinearInterpolation((base, base), reshape(yd, 6, 6))

function itpvf(x)
    return [itp1(x[1], x[2]), itp2(x[1], x[2])]
end

ipprob = ODEProblem((u, p, t) -> itpvf(u), [0.6, -0.6], (0, 30.0))
sol = solve(ipprob; saveat=0.1)

xline = Observable(getindex.(sol.u, 1))
yline = Observable(getindex.(sol.u, 2))

# lines(xarr, yarr)
arrows(getindex.(x, 1), getindex.(x, 2), xarr, yarr)
lines!(xline, yline)

###
## updating the plot
for i in 1:(length(sxdfv) - 1)
    for l in range(0, 1; length=36)
        xd = getindex.(sxdfv[i], 1) * (1 - l) + getindex.(sxdfv[i + 1], 1) * l
        yd = getindex.(sxdfv[i], 2) * (1 - l) + getindex.(sxdfv[i + 1], 2) * l

        itp1 = linear_interpolation((base, base), reshape(xd, 6, 6); extrapolation_bc=0)
        itp2 = linear_interpolation((base, base), reshape(yd, 6, 6); extrapolation_bc=0)

        function itpvf(x)
            return [itp1(x[1], x[2]), itp2(x[1], x[2])]
        end

        ipprob = ODEProblem((u, p, t) -> itpvf(u), [0.6, -0.6], (0, 30.0))
        sol = solve(ipprob; saveat=0.1)

        xline[] = getindex.(sol.u, 1)
        yline[] = getindex.(sol.u, 2)

        xarr[] = xd
        yarr[] = yd
        sleep(0.1)
    end
    println("done with $i")
end
# xarr[] = getindex.(sxdfv[3], 1)
# yarr[] = getindex.(sxdfv[3], 2)

## Interpolation
itp1 = LinearInterpolation((base, base), reshape(getindex.(xd, 1), 6, 6))
itp2 = LinearInterpolation((base, base), reshape(getindex.(xd, 2), 6, 6))

function itpvf(x)
    return [itp1(x[1], x[2]), itp2(x[1], x[2])]
end

begin
    xs = (rand(2) .- 0.5) .* 2
    xds = itpvf(xs) ./ sf
    arrows!([xs[1]], [xs[2]], [xds[1]], [xds[2]]; color=:red)
end