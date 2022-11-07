"""
    DerivativeGP

The Gaussian Process (GP) 
"""
struct DerivativeGP{Tm<:MeanFunction,Tk<:Kernel} <: AbstractGP
    mean_derivative::Tm
    d01_kernel::Tk01
    d10_kernel::Tk10
    d11_kernel::Tk11
end