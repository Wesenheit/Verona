using BenchmarkTools
using CairoMakie
using Verona

eos = Verona.EosTypes.Polytrope(4.0/3.0)
N = 256
P = Verona1D.ParVector1D{Float64,N}()
for i = 1:div(N, 2)
    P.arr1[i] = 1.0
    P.arr2[i] = 10^4/(eos.gamma-1)
end
for i = div(N, 2):N
    P.arr1[i] = 0.1
    P.arr2[i] = 10.0/(eos.gamma-1)
end


X = LinRange(-0.5, 0.5, N) |> collect

dx::Float64 = X[2]-X[1]
dt::Float64 = 0.2*dx
println("CFL for c: ", dt/dx)
T::Float64 = 0.5
n_it::Int64 = 40
tol::Float64 = 1e-8
drops::Float64 = T/3.0
out = Verona1D.LaxFriedrich(P, N, dt, dx, T, eos, drops, n_it, tol)


X = LinRange(-0.5, 0.5, N) |> collect
f = Figure()
ax = Axis(f[1, 1], title = L"$\rho$")
for i = 1:length(out)
    lines!(
        ax,
        X,
        out[i].arr1 |> collect,
        label = "T = " * string(round((i-1)*drops, sigdigits = 2)),
    )
end
f[1, 2] = Legend(f, ax, framevisible = false)
save("LaxFrid_simple.pdf", f)
