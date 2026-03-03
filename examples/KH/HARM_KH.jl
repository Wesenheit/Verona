using BenchmarkTools
using CairoMakie
using ThreadPinning
using Profile
using Printf
using Verona
using MPI
using Random


Type = Float32

MPI.Init()
comm = MPI.COMM_WORLD
MPI_X = 1
MPI_Y = 1
MPI_Z = 1
comm = MPI.Cart_create(comm,(MPI_X,MPI_Y,MPI_Z), periodic=(true,true,false))
Nx = 2048 - 6
Ny = 2048 - 6
Nz = 16 - 6


P = Verona3D.ParVector3D{Type}(Nx, Ny, Nz)
eos = Verona.EosTypes.Polytrope{Type}(4.0/3.0)
tot_X = MPI_X * Nx + 6
tot_Y = MPI_Y * Ny + 6
tot_Z = MPI_Z * Nz + 6

idx, idy, idz = MPI.Cart_coords(comm)
seed = 42
const thread_rngs = [MersenneTwister(seed + i) for i = 1:Threads.nthreads()]
start_calc = time()

Rho0 = 1.
U0 = 1.0
uinf = 0.3
ratio = 0.01

Threads.@threads for num = 1:(P.size_X*P.size_Y*P.size_Z)
    cart_idx = CartesianIndices((P.size_X, P.size_Y, P.size_Z))[num]
    i, j, k = Tuple(cart_idx)
    i_g = Verona.local_to_global(i, idx, P.size_X, MPI_X)
    j_g = Verona.local_to_global(j, idy, P.size_Y, MPI_Y)
    k_g = Verona.local_to_global(k, idz, P.size_Z, MPI_Z)
    P.arr[1,i,j,k] = Rho0
    P.arr[5,i,j,k] = U0
    P.arr[3,i,j,k] = 0.
    P.arr[4,i,j,k] = 0.
    if j_g > div(tot_Y,2)+1
        P.arr[2,i,j,k] = uinf * tanh( (div(tot_Y,4)*3 - j_g) / (ratio * tot_Y))
    else
        P.arr[2,i,j,k] = uinf * tanh( -(div(tot_Y,4) - j_g) / (ratio * tot_Y))
    end
end
P.arr[3,4:end-3,4:end-3,:] .+= reshape(randn(Nx,Ny),Nx,Ny,1) * uinf * 0.1

dx::Type = 1/Nx
dy::Type = 1/Ny
dz::Type = 1/Nz
Cmax::Type = 0.7
dt::Type = Cmax / (1/dx + 1/dy)

T::Type = 10.
drops::Type = T / 100.0

SizeX = 4
SizeY = 4
SizeZ = 4

CuP = Verona3D.CuParVector3D{Type}(P)
n_it::Int64 = 50
tol::Type = 1e-6
floor::Type = 1e-8

Verona3D.HARM_HLL(
    comm,
    CuP,
    (MPI_X, MPI_Y, MPI_Z),
    (SizeX, SizeY, SizeZ),
    dt,
    dx,
    dy,
    dz,
    T,
    eos,
    drops,
    floor,
    ARGS[1],
    true,
    n_it,
    tol,
)

MPI.Finalize():
