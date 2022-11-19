import SciMLBase: AbstractDiffEqFunction

# """
# $(TYPEDEF)
# """
abstract type AbstractGPFunction{iip} <: AbstractDiffEqFunction{iip} end