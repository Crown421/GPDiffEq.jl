# GPDiffEq.jl

The **GPDiffEq.jl** package facilitates learning the (nonlinear) vector field of unknown system using Gaussian Processes (GPs). 

It bridges the Universal Differential Equations in the [SciML](https://sciml.ai/) community using Neural Networks and GP ecosystem by the [JuliaGaussianProcesses organization](https://github.com/JuliaGaussianProcesses/).

For the GP machinery, this package builds on 
- [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl)
- [ApproximateGPs.jl](https://github.com/JuliaGaussianProcesses/ApproximateGPs.jl)
- [InducingPoints.jl](https://github.com/JuliaGaussianProcesses/InducingPoints.jl).

It further uses [Flux.jl](https://github.com/FluxML/Flux.jl) for training, and [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) for solvers. 
