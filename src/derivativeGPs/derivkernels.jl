abstract type AbstractDerivativeKernel <: Kernel end

@doc raw"""
    Derivative10Kernel(kernel)
Derivative of `kernel` with respect to the first argument:

``D^{(1,0)} k(x,y) = \frac{\partial}{\partial x} k(x,y)``

Can be evaluated like a normal kernel.
"""
struct Derivative10Kernel{K} <: AbstractDerivativeKernel
    ker::K
end

function (d::Derivative10Kernel)(
    x::Union{Real,Tuple{Real,Int}}, y::Union{Real,Tuple{Real,Int}}
)
    dx = autodiff(Reverse, d.ker, Active, Active(x), Const(y))
    return first(first(only(dx)))
end
function (d::Derivative10Kernel)(x::AbstractVector, y::AbstractVector)
    dx = zero(x)
    autodiff(Reverse, d.ker, Active, Duplicated(x, dx), Const(y))
    return dx
end
function (d::Derivative10Kernel)(
    (x, px)::Tuple{AbstractVector,Int}, (y, py)::Tuple{AbstractVector,Int}
)
    x = collect(x)
    y = collect(y)
    dx = zero(x)
    autodiff(
        Reverse, (x, y) -> d.ker((x, px), (y, py)), Active, Duplicated(x, dx), Const(y)
    )
    return dx
end

@doc raw"""
    Derivative01Kernel(kernel)
Derivative of `kernel` with respect to the second argument:

``D^{(0,1)} k(x,y) = \frac{\partial}{\partial y} k(x,y)``

Can be evaluated like a normal kernel.
"""
struct Derivative01Kernel{K} <: AbstractDerivativeKernel
    ker::K
end

function (d::Derivative01Kernel)(
    x::Union{Real,Tuple{Real,Int}}, y::Union{Real,Tuple{Real,Int}}
)
    dy = autodiff(Reverse, d.ker, Active, Const(x), Active(y))
    return first(last(only(dy)))
end
function (d::Derivative01Kernel)(x::AbstractVector, y::AbstractVector)
    dy = zero(y)
    autodiff(Reverse, d.ker, Active, Const(x), Duplicated(y, dy))
    return dy
end
function (d::Derivative01Kernel)(
    (x, px)::Tuple{AbstractVector,Int}, (y, py)::Tuple{AbstractVector,Int}
)
    x = collect(x)
    y = collect(y)
    dy = zero(y)
    autodiff(
        Reverse, (x, y) -> d.ker((x, px), (y, py)), Active, Const(x), Duplicated(y, dy)
    )
    return dy
end

@doc raw"""
    Derivative11Kernel(kernel)
Differentiate `kernel` with respect to both arguments. 

``D^{(1,1)} k(x,y) = \frac{\partial^2}{\partial x \partial y} k(x,y)``
    
Can be evaluated like a normal kernel.
"""
struct Derivative11Kernel{K} <: AbstractDerivativeKernel
    ker::K
end

function (d::Derivative11Kernel)(x::Real, y::Real)
    dxy = autodiff(
        Forward,
        (x, y) ->
            first(only(autodiff_deferred(Reverse, d.ker, Active, Active(x), Const(y)))),
        DuplicatedNoNeed,
        Const(x),
        DuplicatedNoNeed(y, 1.0),
    )
    return first(dxy)
end

function (d::Derivative11Kernel)((x, px)::Tuple{Real,Int}, (y, py)::Tuple{Real,Int})
    dxy = autodiff(
        Forward,
        (x, y) -> first(
            only(
                autodiff_deferred(
                    Reverse,
                    (x, y) -> d.ker((x, px), (y, py)),
                    Active,
                    Active(x),
                    Const(y),
                ),
            ),
        ),
        DuplicatedNoNeed,
        Const(x),
        DuplicatedNoNeed(y, 1.0),
    )
    return first(dxy)
end

function (d::Derivative11Kernel)(x::AbstractVector, y::AbstractVector)
    inner = function (x, y)
        dx = zero(x)
        autodiff_deferred(Reverse, d.ker, Active, Duplicated(x, dx), Const(y))
        return dx
    end

    dxyc = [
        only(autodiff(Forward, inner, DuplicatedNoNeed, Const(x), Duplicated(y, dy))) for
        dy in onehot(y)
    ]

    return reduce(hcat, dxyc)
end

function (d::Derivative11Kernel)(
    (x, px)::Tuple{AbstractVector,Int}, (y, py)::Tuple{AbstractVector,Int}
)
    x = collect(x)
    y = collect(y)
    inner = function (x, y)
        dx = zero(x)
        autodiff_deferred(
            Reverse,
            (x, y) -> d.ker((x, px), (y, py)),
            Active,
            Duplicated(x, dx),
            Const(y),
        )
        return dx
    end
    dxyc = [
        only(autodiff(Forward, inner, DuplicatedNoNeed, Const(x), Duplicated(y, dy))) for
        dy in onehot(y)
    ]

    return reduce(hcat, dxyc)
end

@doc raw"""
    DerivativeKernelCollection(kernel)
"""
struct DerivativeKernelCollection{Tk<:Kernel}
    d10::Derivative10Kernel{Tk}
    d01::Derivative01Kernel{Tk}
    d11::Derivative11Kernel{Tk}

    function DerivativeKernelCollection(kernel)
        d10 = Derivative10Kernel(kernel)
        d01 = Derivative01Kernel(kernel)
        d11 = Derivative11Kernel(kernel)
        return new{typeof(kernel)}(d10, d01, d11)
    end
end

#  perhaps not super clear, but this is probably the most often needed part
function (dk::DerivativeKernelCollection)(t1, t2)
    return dk.d10(t1, t2)
end

# need tests, for various kernels and maybe a finitediff method to compare?