# Theory

This packages uses GPs, .... link to basic theory. 

## Modelling choices
While GP models are sometimes referred to as non-parametric, there are some modelling choices that need to be made. 

### Kernel
GPs represent a distribution over a function space, constrained by the available data points.
The choice of kernel determines the Reproducing Kernel Hilbert Space (RKHS) underpinning the distribution, and therefore influences the accuracy of the model uncertainty behaviour. 
 <!-- something space? not index space, but something else  -->

 This package uses [KernelFunction.jl](link), which means that all kernels implemented there are available. 

 ### Inducing Points
 To limit the computational cost, we use inducing points by default. 

 Something, Approximate GPs. 
 