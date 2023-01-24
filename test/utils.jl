using GPDiffEq
using KernelFunctions
using LinearAlgebra
using Test

## reshaping output data
# setup
mker = IndependentMOKernel(SqExponentialKernel())
x = [rand(2) for i in 1:5]
y = ColVecs(rand(2, 5))

# IsotopicByFeatures by hand
xMO1 = KernelFunctions.MOInputIsotopicByFeatures(x, 2)
y1MO = y.X[:]

gp1 = GP(mker)
fgp1 = gp1(xMO1, 0.0001)
pgp1 = posterior(fgp1, y1MO)
my = mean(pgp1, xMO1)
y1 = reshape_isotopic_multi_output(my, pgp1)

@test norm(y1MO - my) < 0.1

@test norm(y.X - y1.X) < 0.1

# IsotopicByOutputs by hand
xMO2 = KernelFunctions.MOInputIsotopicByOutputs(x, 2)
y2MO = permutedims(y.X)[:]

gp2 = GP(mker)
fgp2 = gp2(xMO2, 0.0001)
pgp2 = posterior(fgp2, y2MO)
my = mean(pgp2, xMO2)
y2 = reshape_isotopic_multi_output(my, pgp2)

# y2r = permutedims(reshape(y, 5, 2))
@test norm(y2MO - my) < 0.1
@test norm(y.X - y2.X) < 0.1

# using helper function
xMO3, yMO = prepare_isotopic_multi_output_data(x, y)

gp3 = GP(mker)
fgp3 = gp3(xMO3, 0.0001)
pgp3 = posterior(fgp3, yMO)
my = mean(pgp3, xMO3)
y3 = reshape_isotopic_multi_output(my, pgp3)

@test norm(y3.X - y.X) < 0.1