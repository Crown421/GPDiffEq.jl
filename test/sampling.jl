using AbstractGPs
using GPDiffEq
using Interpolations
using LinearAlgebra
using Test
### 1D example

# setup
x = sort(rand(5))
y = rand(5)
ker = SqExponentialKernel()
gp1 = GP(ker)
fgp1 = gp1(x, 0.01)
pgp1 = posterior(fgp1, y)

# sample
xrange1 = range(0, 1; length=10)

s1 = sample_points(pgp1, xrange1)
f1 = sample_function(pgp1, xrange1)

# test sample points
s1v = [sample_points(pgp1, xrange1) for _ in 1:10000]
s1vm = mean(s1v)

xr1 = collect(xrange1)
tmpy1 = mean(pgp1, xr1)

@test norm(norm.(s1vm .- s1)) > 0.01
@test norm(norm.(s1vm .- tmpy1)) < 0.01

# test hidden function sample function
s1itp = GPDiffEq._create_itps(xrange1, s1, BSpline(Quadratic(Line(OnGrid()))), Line())

@test norm(norm.(s1 .- s1itp.(xr1))) < 1e-10

# test sample function

f1s = [sample_function(pgp1, xrange1).(xr1) for _ in 1:10000]
f1sm = mean(f1s)

@test norm(norm.(f1sm .- tmpy1)) < 0.1

# # plot
# xp = range(0, 1; length=100)
# begin
#     p = plot(xp, mean(pgp1, xp); label="mean", linewidth=3)
#     for i in 1:5
#         f1 = sample_function(pgp1, xrange1)
#         plot!(p, xp, f1.(xp); label="sample function")
#     end
#     p
# end

# need to add tests similarly to the next one. 

### 2D example

# should this work? I would argue yes
# x = [rand(2) for i in 1:5]
# y = [rand(2) for i in 1:5]
x = ColVecs(rand(2, 5))
y = ColVecs(rand(2, 5))
xMO, yMO = prepare_isotopic_multi_output_data(x, y)
mker = IndependentMOKernel(SqExponentialKernel())

gp2 = GP(mker)
fgp2 = gp2(xMO, 0.001)
pgp2 = posterior(fgp2, yMO)

# sample
xl = [
    (minimum(getindex.(x, 1)), maximum(getindex.(x, 1))),
    (minimum(getindex.(x, 2)), maximum(getindex.(x, 2))),
]
nsamples = 10
xrange2 = [range(xl[i]...; length=nsamples) for i in 1:2]

### check sample points accuracy
# single sample for testing
s2 = sample_points(pgp2, xrange2)

# mean for accuracy
# first the GP mean
xr = collect.(Iterators.product(xrange2...))[:]
xr_mo = KernelFunctions.MOInputIsotopicByFeatures(xr, pgp2.data.x.out_dim)
tmpy = mean(pgp2, xr_mo)
testy = reshape_isotopic_multi_output(tmpy, pgp2)

# now sample a bunch of sample points, take the mean and compare
s2v = [sample_points(pgp2, xrange2) for _ in 1:10000]
s2vm = mean(s2v)

@test norm(norm.(s2vm .- s2)) > 0.1
@test norm(norm.(s2vm .- testy)) < 0.1

### check sample function accuracy on grid points
# underlying function
# single function matches the samples points
s2itp = GPDiffEq._create_itps(xrange2, s2, BSpline(Quadratic(Line(OnGrid()))), Line())

@test norm(norm.(s2 .- s2itp.(xr))) < 1e-10

# larger sample of functions 
f2s = [sample_function(pgp2, xrange2).(xr) for _ in 1:5000]
f2sm = mean(f2s)
@test norm(norm.(f2sm .- testy)) < 0.1

# complete pipeline
# single sample for testing

f2 = sample_function(pgp2, xrange2)

### check sample function accuracy outside gridpoints
# mean for accuracty
nevals = 14
# xp = [range(xl[i]...; length=nevals) for i in 1:2]
# xp = collect.(Iterators.product(xp...))[:]
# xp = xr
# xp = x
xp = [rand(2) for i in 1:nevals]

function transform(x, xl)
    x = x .* (getindex.(xl, 2) - getindex.(xl, 1)) .+ getindex.(xl, 1)
    return x
end

xp = transform.(xp, Ref(xl))

xpMO = KernelFunctions.MOInputIsotopicByFeatures(xp, 2)
m = mean(pgp2, xpMO)
m = reshape_isotopic_multi_output(m, pgp2)

msv = [sample_function(pgp2, xrange2).(xp) for _ in 1:5000]
msvm = mean(msv)

@test norm(norm.(msvm .- m)) < 0.1

# ToDo: Add test with and without default args
