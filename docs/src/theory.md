# Theory

This packages uses GPs to learn the vector field of unknown dynamical systems. The rationale is that for many systems, the vector field is substantially simpler than the flow map (see e.g. [emergence](https://www.nbi.dk/~emmeche/coPubl/97d.NABCE/ExplEmer.html))

## Modelling choices
While GP models are sometimes referred to as non-parametric, there are some modelling choices that need to be made. 

### Kernel
GPs represent a distribution over a function space, constrained by the available data points.
The choice of kernel determines the Reproducing Kernel Hilbert Space (RKHS) underpinning the distribution, and therefore influences the accuracy of the model uncertainty behaviour. 

This package uses [KernelFunction.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl), which means that all kernels implemented there are available. 

### Inducing Points
To limit the computational cost, we use inducing points by default. Various methods to select inducing points can be found in the [InducingPoints.jl](https://github.com/JuliaGaussianProcesses/InducingPoints.jl) package.


 