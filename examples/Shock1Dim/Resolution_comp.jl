using BenchmarkTools
using CairoMakie
using Verona

eos = Verona.EosTypes.Polytrope{Float64}(4.0/3.0)

function get_Vector(N::Int64)
    P = Verona1D.ParVector1D{Float64}(N)
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
    return P,dx,dt
end

T::Float64 = 0.5
n_it::Int64 = 40.
tol::Float64 = 1e-6
drops::Float64 = T/3.

methods = [:minmod,:ppm,:wenoz]
sizes = [64,128,256,512,1024,2048,4096,8192,16384]
bench = 32768
glob_err = zeros(length(methods),length(sizes))
for (i,method) in enumerate(methods)
    P,dx,dt = get_Vector(bench)
    out_bench = Verona1D.HARM_HLL(P,bench,dt,dx,T,eos,drops,method,n_it,tol)[end-1]
    for (j,size) in enumerate(sizes)
        P,dx,dt = get_Vector(size)
        out= Verona1D.HARM_HLL(P,size,dt,dx,T,eos,drops,method,n_it,tol)[end-1]
        reduction = div(bench,size)
        out_bench_rho = out_bench.arr1[1:reduction:end]
        err = sqrt( sum( (out.arr1 .- out_bench_rho) .^ 2)/length(out_bench_rho))
        println(size," ",err)
        glob_err[i,j] = err
    end
end

f = Figure()
ax = Axis(f[1, 1],xlabel = "N", ylabel = "RMSE error",xscale = log2,yscale = log10, title = L"$\delta \rho$")
names = ["Minmod","PPM","WenoZ"]
dxs = 1 ./ sizes
for i in 1:length(methods)
    lines!(ax,sizes,glob_err[i,:],label = names[i])
end
f[1, 2] = Legend(f, ax, framevisible = false)
save("Comp_res.pdf",f)


