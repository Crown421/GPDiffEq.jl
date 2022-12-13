import SciMLBase: AbstractODEFunction

export GPODEFunction

"""
$(TYPEDEF)
"""
abstract type AbstractGPODEFunction{iip} <: AbstractODEFunction{iip} end

# @doc """

# Let's see what we need
# - thoughts:
#     - f is GP
#     - jac (kind of?) id DerivGP, might need some convenience function to get jacobian of mean. Maybe that happens when constructing a problem, based on the solver? or, wait, I will need 
# """
struct GPODEFunction{iip,GP,DGP} <: AbstractGPODEFunction{iip}
    gp::GP
    dgp::DGP
    # jac

    function GPODEFunction{iip}(
        gp::AbstractGPs.AbstractGP, dgp::AbstractGPs.AbstractGP
    ) where {iip}
        return new{iip,typeof(gp),typeof(dgp)}(gp, dgp)
    end
end

## ToDo: paramters at some point
# (gpf::GPODEFunction)(x, p, t) = _mean(gpf.gp, x)
(gpf::GPODEFunction)(x) = _mean(gpf.gp, x)