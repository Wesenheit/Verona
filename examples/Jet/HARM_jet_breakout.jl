using BenchmarkTools
using CUDA
using MPI
using Verona

Type = Float64

@assert MPI.has_cuda()
MPI.Init()
comm = MPI.COMM_WORLD
MPI_X = 1
MPI_Y = 1
comm = MPI.Cart_create(comm,(MPI_X,MPI_Y), periodic=(false,false),reorder = true)

eos = Verona.EosTypes.Polytrope{Type}(4.0/3.0)
Nx = 2048 - 6
Ny = 2048 - 6
P = Verona2D.ParVector2D{Type}(Nx,Ny)
tot_X = MPI_X * Nx + 6
tot_Y = MPI_Y * Ny + 6

idx,idy=  MPI.Cart_coords(comm)

#one unit -> 10^8cm

floor::Type = 1e-20
outer::Type = 1e-5
box_X::Type = 200.
box_Y::Type = 800.
R_max::Type = 100.
R_eng::Type = 5.
Rho0::Type = 2.5e-2
Temp::Type = 1e-4
U0::Type = 2.5e-1
dx::Type = 2*box_X / (tot_X)
dy::Type = box_Y/tot_Y
Gamma0::Type = 10.
for i in 1:P.size_X
    for j in 1:P.size_Y
        i_g = Verona2D.local_to_global(i,idx,P.size_X,MPI_X)
        j_g = Verona2D.local_to_global(j,idy,P.size_Y,MPI_Y)
        if i_g == 0 || j_g == 0 
            continue
        end
        X = i_g * dx - box_X
        Y = j_g * dy
        R = sqrt(X^2+Y^2)
        if R < R_max
            if R < R_eng
                rho = Rho0 * (R_max/R_eng)^2 * (1+randn()*1e-2)
            else
                rho = Rho0 * (R_max/R)^2 * (1+randn()*1e-2)
            end
        else
            rho = outer
        end
        P.arr[1,i,j] = rho
        P.arr[2,i,j] = rho*Temp

        angle = atan(Y,X)
        if R < R_eng
            v = 1/(1-1/Gamma0^2) * 1/cosh(X/R_eng)^8
            uy = Gamma0 * v * (1+randn()*1e-2)
            ux = 0
            P.arr[1,i,j] = rho*1e-3
            P.arr[2,i,j] = P.arr[1,i,j]*1e-4
        else
            ux = 0
            uy = 0
        end
        P.arr[3,i,j] = ux
        P.arr[4,i,j] = uy
    end
end
Cmax::Type = 0.8
dt::Type = Cmax /(1/dx + 1/dy)
T::Type = box_Y
n_it::Int64 = 50.
tol::Type = 1e-6

T_exp::Type = box_Y/10

function TurnOff(P,t)
    if t > box_Y/3 && t < box_Y
        P.arr[4,:,1:3] .*=  (1-dt/T_exp)
    end
    P.arr[1:4,1,:] .= P.arr[1:4,4,:] 
    P.arr[1:4,2,:] .= P.arr[1:4,4,:] 
    P.arr[1:4,3,:] .= P.arr[1:4,4,:] 

    P.arr[1:4,end-2,:] .= P.arr[1:4,end-3,:] 
    P.arr[1:4,end-1,:] .= P.arr[1:4,end-3,:] 
    P.arr[1:4,end,:] .= P.arr[1:4,end-3,:] 
    
    P.arr[1:4,:,end-2] .= P.arr[1:4,:,end-3] 
    P.arr[1:4,:,end-1] .= P.arr[1:4,:,end-3] 
    P.arr[1:4,:,end] .= P.arr[1:4,:,end-3] 
end

if MPI.Comm_rank(comm) == 0
    println("dt: ",dt)
end
drops::Type = T/100.
SizeX = 16
SizeY = 16
CuP = Verona2D.CuParVector2D{Type}(P)
CUDA.@time Verona2D.HARM_HLL(comm,CuP,MPI_X,MPI_Y,SizeX,SizeY,dt,dx,dy,T,eos,drops,floor,ARGS[1],n_it,tol,TurnOff)

MPI.Finalize()
