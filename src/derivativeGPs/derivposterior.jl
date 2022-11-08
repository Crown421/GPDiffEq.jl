
# need to construct with base, undifferentiated GP
function posterior(dfx::AbstractGPs.FiniteGP{<:DerivativeGP}, y::AbstractVector{<:Real})
    m = mean(dfx.f.f, dfx.x)
    C_mat = cov(dfx.f.f, dfx.x) + dfx.Σy
    C = cholesky(AbstractGPs._symmetric(C_mat))
    δ = y - m
    α = C \ δ
    return AbstractGPs.PosteriorGP(dfx.f, (α=α, C=C, x=dfx.x, δ=δ))
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
    C_xcond_x = f.prior.dkernel.d10.(f.data.x, x')
    C_xcond_y = f.prior.dkernel.d01.(f.data.x, z')
    return cov(f.prior, x, z) - AbstractGPs.Xt_invA_Y(C_xcond_x, f.data.C, C_xcond_y)
end

# function StatsBase.mean_and_cov(
#     f::AbstractGPs.PosteriorGP{<:DerivativeGP}, x::AbstractVector
# )
#     C_xcond_x = cov(f.prior, f.data.x, x)
#     m_post = mean(f.prior, x) + C_xcond_x' * f.data.α
#     C_post = cov(f.prior, x) - Xt_invA_X(f.data.C, C_xcond_x)
#     return (m_post, C_post)
# end

# function StatsBase.mean_and_var(
#     f::AbstractGPs.PosteriorGP{<:DerivativeGP}, x::AbstractVector
# )
#     C_xcond_x = cov(f.prior, f.data.x, x)
#     m_post = mean(f.prior, x) + C_xcond_x' * f.data.α
#     C_post_diag = var(f.prior, x) - diag_Xt_invA_X(f.data.C, C_xcond_x)
#     return (m_post, C_post_diag)
# end
