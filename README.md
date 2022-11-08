# GPDiffEq

[![julia 1.8](https://github.com/Crown421/GPDiffEq.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/Crown421/GPDiffEq.jl/actions/workflows/ci.yml) 
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://crown421.github.io/GPDiffEq.jl/dev/)
[![Coverage Status](https://coveralls.io/repos/github/Crown421/GPDiffEq.jl/badge.svg?branch=main)](https://coveralls.io/github/Crown421/GPDiffEq.jl?branch=main)


This package is intended to connect Gaussian Processes and differential equations in a similar fashion to [DiffEqFlux.jl](https://github.com/SciML/DiffEqFlux.jl). 

It is the successor of [GaussianProcessODEs.jl](https://github.com/Crown421/GaussianProcessODEs.jl), which contains mostly original implementations and minimally documented research code. By contrast the `GPDiffEq` packages aims to take full advantage of the broader [JuliaGaussianProcesses](https://juliagaussianprocesses.github.io/) eco system. 

The motto of this package is currently "End-to-end, ready to be fixed". The basic functionality exists and now needs to be made robust and extended. 