var documenterSearchIndex = {"docs":
[{"location":"api/#GPDiffEq-API","page":"GPDiffEq API","title":"GPDiffEq API","text":"","category":"section"},{"location":"api/","page":"GPDiffEq API","title":"GPDiffEq API","text":"Modules = [GPDiffEq]","category":"page"},{"location":"theory/#Theory","page":"Theory","title":"Theory","text":"","category":"section"},{"location":"theory/","page":"Theory","title":"Theory","text":"This packages uses GPs to learn the vector field of unknown dynamical systems. The rationale is that for many systems, the vector field is substantially simpler than the flow map (see e.g. emergence)","category":"page"},{"location":"theory/#Modelling-choices","page":"Theory","title":"Modelling choices","text":"","category":"section"},{"location":"theory/","page":"Theory","title":"Theory","text":"While GP models are sometimes referred to as non-parametric, there are some modelling choices that need to be made. ","category":"page"},{"location":"theory/#Kernel","page":"Theory","title":"Kernel","text":"","category":"section"},{"location":"theory/","page":"Theory","title":"Theory","text":"GPs represent a distribution over a function space, constrained by the available data points. The choice of kernel determines the Reproducing Kernel Hilbert Space (RKHS) underpinning the distribution, and therefore influences the accuracy of the model uncertainty behaviour. ","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"This package uses KernelFunction.jl, which means that all kernels implemented there are available. ","category":"page"},{"location":"theory/#Inducing-Points","page":"Theory","title":"Inducing Points","text":"","category":"section"},{"location":"theory/","page":"Theory","title":"Theory","text":"To limit the computational cost, we use inducing points by default. Various methods to select inducing points can be found in the InducingPoints.jl package.","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"","category":"page"},{"location":"#GPDiffEq.jl","page":"GPDiffEq.jl","title":"GPDiffEq.jl","text":"","category":"section"},{"location":"","page":"GPDiffEq.jl","title":"GPDiffEq.jl","text":"The GPDiffEq.jl package facilitates learning the (nonlinear) vector field of unknown system using Gaussian Processes (GPs). ","category":"page"},{"location":"","page":"GPDiffEq.jl","title":"GPDiffEq.jl","text":"It bridges the Universal Differential Equations in the SciML community using Neural Networks and GP ecosystem by the JuliaGaussianProcesses organization.","category":"page"},{"location":"","page":"GPDiffEq.jl","title":"GPDiffEq.jl","text":"For the GP machinery, this package builds on ","category":"page"},{"location":"","page":"GPDiffEq.jl","title":"GPDiffEq.jl","text":"KernelFunctions.jl\nApproximateGPs.jl\nInducingPoints.jl.","category":"page"},{"location":"","page":"GPDiffEq.jl","title":"GPDiffEq.jl","text":"It further uses Flux.jl for training, and DifferentialEquations.jl for solvers. ","category":"page"},{"location":"symmetries/#Symmetries","page":"Symmetries","title":"Symmetries","text":"","category":"section"},{"location":"symmetries/","page":"Symmetries","title":"Symmetries","text":"In addition to the kernels from KernelFunctions, we implement Group-Integration-Matrix (GIM) Kernels. ","category":"page"}]
}
