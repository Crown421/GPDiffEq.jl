# import ApproximateGPs: posterior
export differentiate

"""
    posterior(FiniteGP{<:DerivativeGP}, y::AbstractVector{<:Real})
The posterior of a derivative GP, conditioned on the data `y` from the output space of the undifferentiated GP. Evaluating this posterior at a point `x` will return the posterior of the derivative at `x`, and therefore not return the original data `y`.
"""
function AbstractGPs.posterior(
    dfx::AbstractGPs.FiniteGP{<:DerivativeGP}, y::AbstractVector{<:Real}
)
    # need to construct C with base, undifferentiated GP
    # could add a method here for finite GPs, that computes the right mean_and_cov of this specific case
    m = mean(dfx.f.f, dfx.x)
    C_mat = cov(dfx.f.f, dfx.x) + dfx.Σy
    C = cholesky(AbstractGPs._symmetric(C_mat))
    δ = y - m
    α = C \ δ
    return AbstractGPs.PosteriorGP(dfx.f, (α=α, C=C, x=dfx.x, δ=δ))
end

"""
    posterior(fx::FiniteGP{<:PosteriorGP}, y::AbstractVector{<:Real})
Construct the posterior distribution over `fx.f` when `f` is itself a `PosteriorGP` by
updating the Cholesky factorisation of the covariance matrix and avoiding recomputing it
from the original covariance matrix. It does this by using `update_chol` functionality.
Other aspects are similar to a regular posterior.
"""
function AbstractGPs.posterior(
    fx::AbstractGPs.FiniteGP{<:AbstractGPs.PosteriorGP{<:DerivativeGP}},
    y::AbstractVector{<:Real},
)
    m2 = mean(fx.f.prior, fx.x)
    δ2 = y - m2
    C12 = cov(fx.f.prior.f, fx.f.data.x, fx.x)
    C22 = cov(fx.f.prior.f, fx.x) + fx.Σy
    chol = AbstractGPs.update_chol(fx.f.data.C, C12, C22)
    δ = vcat(fx.f.data.δ, δ2)
    α = chol \ δ
    x = vcat(fx.f.data.x, fx.x)
    return AbstractGPs.PosteriorGP(fx.f.prior, (α=α, C=chol, x=x, δ=δ))
end

### AbstractGP interface implementation.
# PosteriorGP not exported seems odd. Probably should import explicitly.
function Statistics.mean(f::AbstractGPs.PosteriorGP{<:DerivativeGP}, x::AbstractVector)
    return mean(f.prior, x) + f.prior.dkernel.d10.(x, f.data.x') * f.data.α
end

function Statistics.cov(f::AbstractGPs.PosteriorGP{<:DerivativeGP}, x::AbstractVector)
    #ToDo: Need test for correctness
    return cov(f.prior, x) -
           AbstractGPs.Xt_invA_X(f.data.C, f.prior.dkernel.d10.(x', f.data.x))
end

function Statistics.var(f::AbstractGPs.PosteriorGP{<:DerivativeGP}, x::AbstractVector)
    return var(f.prior, x) -
           AbstractGPs.diag_Xt_invA_X(f.data.C, f.prior.dkernel.d10.(x', f.data.x))
end

function Statistics.cov(
    f::AbstractGPs.PosteriorGP{<:DerivativeGP}, x::AbstractVector, z::AbstractVector
)
    C_xcond_x = f.prior.dkernel.d10.(x', f.data.x)
    C_xcond_y = f.prior.dkernel.d01.(f.data.x, z')
    return cov(f.prior, x, z) - AbstractGPs.Xt_invA_Y(C_xcond_x, f.data.C, C_xcond_y)
end

function StatsBase.mean_and_cov(
    f::AbstractGPs.PosteriorGP{<:DerivativeGP}, x::AbstractVector
)
    C_xcond_x = f.prior.dkernel.d10.(x', f.data.x)
    m_post = mean(f.prior, x) + C_xcond_x' * f.data.α
    C_post = cov(f.prior, x) - AbstractGPs.Xt_invA_X(f.data.C, C_xcond_x)
    return (m_post, C_post)
end

function StatsBase.mean_and_var(
    f::AbstractGPs.PosteriorGP{<:DerivativeGP}, x::AbstractVector
)
    C_xcond_x = f.prior.dkernel.d10.(x', f.data.x)
    m_post = mean(f.prior, x) + C_xcond_x' * f.data.α
    C_post_diag = var(f.prior, x) - AbstractGPs.diag_Xt_invA_X(f.data.C, C_xcond_x)
    return (m_post, C_post_diag)
end

## Conversion methods
# ToDo: need tests
"""
    differentiate(gpp::PosteriorGP)
"""
function differentiate(gpp::AbstractGPs.PosteriorGP)
    dgp = DerivativeGP(gpp.prior)
    return AbstractGPs.PosteriorGP(dgp, gpp.data)
end
