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
MPI_Z = 1
comm = MPI.Cart_create(comm,(MPI_X,MPI_Y,MPI_Z), periodic=(false,false,false),reorder = true)

eos = Verona.EosTypes.Polytrope{Type}(4.0/3.0)
Nx = 512 - 6
Ny = 256 - 6
Nz = 256 - 6

P = Verona3D.ParVector3D{Type}(Nx,Ny,Nz)
tot_X = MPI_X * Nx + 6
tot_Y = MPI_Y * Ny + 6
tot_Z = MPI_Z * Nz + 6

idx,idy,idz=  MPI.Cart_coords(comm)

#one unit -> 10^8cm

floor::Type = 1e-12
outer::Type = 1e-4
box_X::Type = 480.
box_Y::Type = 120.
box_Z::Type = 120.
R_max::Type = 100.
R_eng::Type = 5
Rho0::Type = 2.5e-2
Temp::Type = 1e-5
U0::Type = 2.5e-1

dx::Type = box_X /tot_X
dy::Type = 2*box_Y/tot_Y
dz::Type = 2*box_Z/tot_Z
Gamma0::Type = 10.

for i in 1:P.size_X
    for j in 1:P.size_Y
        for k in 1:P.size_Z
            i_g = Verona3D.local_to_global(i,idx,P.size_X,MPI_X)
            j_g = Verona3D.local_to_global(j,idy,P.size_Y,MPI_Y)
            k_g = Verona3D.local_to_global(k,idz,P.size_Z,MPI_Z)
            if i_g == 0 || j_g == 0 || k_g == 0
                continue
            end
            X = i_g * dx
            Y = j_g * dy - box_Y
            Z = k_g * dz - box_Z

            R = sqrt(X^2 + Y^2 + Z^2)
            if R < R_max
                if R < R_eng
                    rho = Rho0 * (R_max/R_eng)^2 * (1+randn()*3e-2)
                else
                    rho = Rho0 * (R_max/R)^2 * (1+randn()*3e-2)
                end
            else
                rho = outer
            end
            P.arr[1,i,j,k] = rho
            P.arr[2,i,j,k] = rho*Temp

            if R < R_eng
                v = 1/(1-1/Gamma0^2) * 1/cosh( sqrt(X^2+Y^2) /R_eng)^8
                ux = Gamma0 * v
                uy = 0
                uz = 0
                P.arr[1,i,j,k] = rho*1e-2
                P.arr[2,i,j,k] = P.arr[1,i,j,k]*1e-5
            else
                ux = 0
                uy = 0
                uz = 0
            end
            P.arr[3,i,j,k] = ux
            P.arr[4,i,j,k] = uy
            P.arr[5,i,j,k] = uz
        end
    end
end
Cmax::Type = 0.8
dt::Type = Cmax /(1/dx + 1/dy + 1/dz)
T::Type = box_X
n_it::Int64 = 50.
tol::Type = 1e-6

T_exp::Type = box_X/10


function TurnOff(P,t)
    if t > box_X/3 && t < box_X
        #turn off the engine
        P.arr[3,1:3,:,:] .*=  (1-dt/T_exp)
    end
    #zero grad Y
    P.arr[:,:,1,:] .= P.arr[:,:,4,:] 
    P.arr[:,:,2,:] .= P.arr[:,:,4,:] 
    P.arr[:,:,3,:] .= P.arr[:,:,4,:] 

    P.arr[:,:,end-2,:] .= P.arr[:,:,end-3,:] 
    P.arr[:,:,end-1,:] .= P.arr[:,:,end-3,:] 
    P.arr[:,:,end,:] .= P.arr[:,:,end-3,:] 
    
    #zero grad Z
    P.arr[:,:,:,1] .= P.arr[:,:,:,4] 
    P.arr[:,:,:,2] .= P.arr[:,:,:,4] 
    P.arr[:,:,:,3] .= P.arr[:,:,:,4] 

    P.arr[:,:,:,end-2] .= P.arr[:,:,:,end-3] 
    P.arr[:,:,:,end-1] .= P.arr[:,:,:,end-3] 
    P.arr[:,:,:,end] .= P.arr[:,:,:,end-3] 
    
    #zero grad X
    P.arr[:,end-2,:,:] .= P.arr[:,end-3,:,:] 
    P.arr[:,end-1,:,:] .= P.arr[:,end-3,:,:] 
    P.arr[:,end,:,:] .= P.arr[:,end-3,:,:] 
end

if MPI.Comm_rank(comm) == 0
    println("dt: ",dt)
end
drops::Type = T/100.
SizeX = 4
SizeY = 4
SizeZ = 4
CuP = Verona3D.CuParVector3D{Type}(P)
CUDA.@time Verona3D.HARM_HLL(comm,CuP,MPI_X,MPI_Y,MPI_Z,SizeX,SizeY,SizeZ,dt,dx,dy,dz,T,eos,drops,floor,ARGS[1],n_it,tol,TurnOff)

MPI.Finalize()
