# # Derivative of a GP

# The derivative of a GP is also a GP, as differentiation is a linear operators. In this example, we show how to use of the `DerivativeGP` and how to easily generate it from a known GP.  

# ## 1D Example

# ### Setup
using GPDiffEq
using Plots
using LinearAlgebra
using Zygote

# ### The toy model
# We generate data for our model
σ_n = 3e-2
x = collect(range(-3, 3; length=10))
y = sin.(x) + σ_n * randn(length(x))

# which looks as follows
x_plot = collect(range(-3, 3; length=50))

plot(x_plot, sin.(x_plot); label="true", linewidth=2.5)
scatter!(x, y; label="data", markersize=5)
plot!(; legend=:topleft) #hide

# ### Define a GP
# We'll use a simple GP with a `ZeroMean` and `GaussianKernel`, condition it on our data

kernel = GaussianKernel()
f = GP(kernel)
fx = f(x, σ_n^2)

f_post = posterior(fx, y)

# and plot the posterior. Note that this GP completely untrained, no hyperparameters have been defined. 

plot(x_plot, sin.(x_plot); label="true", linewidth=2.5)
scatter!(x, y; label="data", markersize=5)
plot!(
    x_plot,
    mean(f_post, x_plot);
    ribbons=sqrt.(var(f_post, x_plot)),
    label="GP",
    linewidth=2.5,
)
plot!(; legend=:topleft) #hide

# ### Derivative of a GP
# Now we can easily generate the derivate of this GP
df_post = differentiate(f_post)

# and plot the new posterior. As we know, the derivative of `sin` is `cos`, so we can check the differentiated GP.

plot(x_plot, cos.(x_plot); label="true", linewidth=2.5)
plot!(
    x_plot,
    mean(df_post, x_plot);
    ribbons=sqrt.(var(df_post, x_plot)),
    label="GP",
    linewidth=2.5,
)

# As we saw above, the original GP was slighly off from the true function. This is reflected and amplified in the derivative as well.

# ## 2D Example

# We can do the same for a Multi-Input-Multi-Output GP, using the KernelFunctions.jl [multi-output interface](https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/api/#Inputs-for-Multiple-Outputs).

# ### Setup

function fun(x)
    return [-0.1 2.0; -2.0 -0.1] * (x .^ 3)
end
xrange = range(-1, 1; length=4)
x = collect.(Iterators.product(xrange, xrange))[:]
y = fun.(x)
## This is annoying UX, needs fix
y = ColVecs(reduce(hcat, y))
xMO, yMO = prepare_isotopic_multi_output_data(x, y)

# ### Defining a Multi-Output GP
σ_n = 3e-2
ker = GaussianKernel()
mker = IndependentMOKernel(ker)

f = GP(mker)
fx = f(xMO, σ_n)
f_post = posterior(fx, yMO)

# which looks as follows
sf = maximum(norm.(y)) * 2
quiver(
    getindex.(xMO.x, 1),
    getindex.(xMO.x, 2);
    quiver=(y.X[1, :] ./ sf, y.X[2, :] ./ sf),
    label="data",
    markersize=5,
)

# ### Derivative of a Multi-Output GP

# As above, the derivate is obtained very easily using our provided function:
df_post = differentiate(f_post)

# ### Visualizing the derivatives
# To demonstrate, we we show the contour plots for each component of our function and the corresponding (scaled) gradient separately. 
# In the first row of the following grid we show the $f_1$ and in the second row $f_2$. 
# The right column showing the GP also includes the location of the input data points for the GP. 

# As our GP is untrained, it smoothes the data a little too strongly, but we can see that the gradients are correctly perpendicular to the contours. 
xprange = range(-1, 1; length=14)
xprange2 = range(-1, 1.0; length=10)
xp = vcat.(xprange', xprange2)

p = plot(; layout=(2, 2), size=(800, 800))

for comp in 1:2
    nlevels = 30
    narrow = 6

    spl = (comp - 1) * 2
    fungrad(xval) = only(gradient(x -> fun(x)[comp], xval))
    gpgrad(xval) = only(mean(df_post, [(xval, comp)]))
    zreal = getindex.(fun.(xp), comp)

    contour!(
        p,
        xprange,
        xprange2,
        zreal;
        levels=nlevels,
        linewidth=2,
        label="",
        subplot=spl + 1,
        title="real",
    )

    xpMO = [[(xval, comp)] for xval in xp]
    zgp = only.(mean.(Ref(f_post), xpMO))
    contour!(
        p,
        xprange,
        xprange2,
        zgp;
        levels=nlevels,
        linewidth=2,
        label="",
        subplot=spl + 2,
        title="gp",
    )

    scatter!(
        p,
        getindex.(x, 1),
        getindex.(x, 2);
        label="data",
        markersize=3,
        subplot=spl + 2,
        legend=:none,
    )

    xgrange = range(-0.8, 0.8; length=narrow)
    xg = collect.(Iterators.product(xgrange, xgrange))[:]
    dzreal = fungrad.(xg)
    sf = maximum(norm.(dzreal)) * 3

    quiver!(
        p,
        getindex.(xg, 1),
        getindex.(xg, 2);
        quiver=(getindex.(dzreal, 1) ./ sf, getindex.(dzreal, 2) ./ sf),
        label="data",
        linewidth=2.5,
        subplot=spl + 1,
    )

    dzgp = gpgrad.(xg)
    quiver!(
        p,
        getindex.(xg, 1),
        getindex.(xg, 2);
        quiver=(getindex.(dzgp, 1) ./ sf, getindex.(dzgp, 2) ./ sf),
        label="data",
        linewidth=2.5,
        subplot=spl + 2,
    )
end
p
