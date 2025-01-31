using BenchmarkTools
using CairoMakie

using Verona

eos = Verona.EosTypes.Polytrope{Float64}(4.0/3.0)
N = 256
P = Verona1D.ParVector1D{Float64,N}()
for i in 1:div(N,2)
    P.arr1[i] = 1.
    P.arr2[i] = 10^4/(eos.gamma-1)
end
for i in div(N,2):N
    P.arr1[i] = 0.1
    P.arr2[i] = 10.0/(eos.gamma-1)
end

X = LinRange(-0.5,0.5,N) |> collect

dx::Float64 = X[2]-X[1]
dt::Float64 = 0.8 * dx
println("CFL for c: ",dt/dx)
T::Float64 = 0.5
n_it::Int64 = 40.
tol::Float64 = 1e-6
drops::Float64 = T/3.

#three to choose, :ppm, :wenoz and :minmod
method = :wenoz

out = Verona1D.HARM_HLL(P,N,dt,dx,T,eos,drops,method,n_it,tol)


X = LinRange(-0.5,0.5,N) |> collect
f = Figure()
ax_rho = Axis(f[1, 1],xlabel = L"$X$", ylabel = L"$\rho$")
ax_P = Axis(f[1, 2],xlabel = L"$X$",ylabel = L"$p$")
ax_gamm =  Axis(f[2, 1],xlabel = L"$X$",ylabel = L"$\Gamma$")
ax_vel = Axis(f[2,2],xlabel = L"$X$",ylabel = L"$v_X$")
for i in 1:length(out)
    lines!(ax_rho, X, out[i].arr1 |> collect,color = "black")
    lines!(ax_P, X, (eos.gamma -1) * out[i].arr2 |> collect,color = "black")
    gamma = sqrt.( out[i].arr3 .^2 .+ 1)
    lines!(ax_gamm, X, gamma,color = "black")
    lines!(ax_vel, X, out[i].arr3 ./ gamma,color = "black")
end
save("HARM_HLL.pdf",f)


f = Figure()
ax = Axis(f[1, 1],title = L"$\rho$")
for i in 1:length(out)
    lines!(ax, X, out[i].arr1 |> collect,label = "T = " * string(round((i-1)*drops,sigdigits = 2)))
end
f[1, 2] = Legend(f, ax, framevisible = false)
save("HARM_HLL_simple.pdf",f)

