# # Derivative of a GP

# The derivative of a GP is also a GP, as differentiation is a linear operators. In this example, we show how to use of the `DerivativeGP` and how to easily generate it from a known GP.  

# ## Setup
using GPDiffEq
using Plots

# ## The toy model
# We generate data for our model
σ_n = 3e-2
x = collect(range(-3, 3; length=10))
y = sin.(x) + σ_n * randn(length(x))

# which looks as follows
x_plot = collect(range(-3, 3; length=50))

plot(x_plot, sin.(x_plot); label="true", linewidth=2.5)
scatter!(x, y; label="data", markersize=5)
plot!(; legend=:topleft) #hide

## Define a GP
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

# ## Derivative of a GP
# Now we can very easily generate the derivate of this GP
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