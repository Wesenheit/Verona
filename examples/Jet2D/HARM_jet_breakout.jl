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
comm = MPI.Cart_create(comm, (MPI_X, MPI_Y), periodic = (false, false), reorder = true)

eos = Verona.EosTypes.Polytrope{Type}(4.0/3.0)
Nx = 4096 - 6
Ny = 2048 - 6
P = Verona2D.ParVector2D{Type}(Nx, Ny)
tot_X = MPI_X * Nx + 6
tot_Y = MPI_Y * Ny + 6

idx, idy = MPI.Cart_coords(comm)

#one unit -> 10^8cm

floor::Type = 1e-12
outer::Type = 1e-4
box_X::Type = 800.0
box_Y::Type = 200.0
R_max::Type = 100.0
R_eng::Type = 2.0
Rho0::Type = 2.5e-2
Temp::Type = 1e-5
U0::Type = 2.5e-1
dx::Type = box_X / tot_X
dy::Type = 2*box_Y/tot_Y
Gamma0::Type = 10.0
for i = 1:P.size_X
    for j = 1:P.size_Y
        i_g = Verona.local_to_global(i, idx, P.size_X, MPI_X)
        j_g = Verona.local_to_global(j, idy, P.size_Y, MPI_Y)
        if i_g == 0 || j_g == 0
            continue
        end
        X = i_g * dx
        Y = j_g * dy - box_Y
        R = sqrt(X^2+Y^2)
        if R < R_max
            if R < R_eng
                rho = Rho0 * (R_max/R_eng)^2 * (1+randn()*3e-2)
            else
                rho = Rho0 * (R_max/R)^2 * (1+randn()*3e-2)
            end
        else
            rho = outer
        end
        P.arr[1, i, j] = rho
        P.arr[2, i, j] = rho*Temp

        angle = atan(Y, X)
        if R < R_eng
            v = 1/(1-1/Gamma0^2) * 1/cosh(Y/R_eng)^8
            ux = Gamma0 * v
            uy = 0
            P.arr[1, i, j] = rho*1e-3
            P.arr[2, i, j] = P.arr[1, i, j]*1e-5
        else
            ux = 0
            uy = 0
        end
        P.arr[3, i, j] = ux
        P.arr[4, i, j] = uy
    end
end
Cmax::Type = 0.8
dt::Type = Cmax / (1/dx + 1/dy)
T::Type = box_X
n_it::Int64 = 50.0
tol::Type = 1e-6

T_exp::Type = box_X/10


function TurnOff(P, t)
    if t > box_X/3 && t < box_X
        P.arr[4, 1:3, :] .*= (1-dt/T_exp)
    end
    P.arr[1:4, :, 1] .= P.arr[1:4, :, 4]
    P.arr[1:4, :, 2] .= P.arr[1:4, :, 4]
    P.arr[1:4, :, 3] .= P.arr[1:4, :, 4]

    P.arr[1:4, :, end-2] .= P.arr[1:4, :, end-3]
    P.arr[1:4, :, end-1] .= P.arr[1:4, :, end-3]
    P.arr[1:4, :, end] .= P.arr[1:4, :, end-3]

    P.arr[1:4, end-2, :] .= P.arr[1:4, end-3, :]
    P.arr[1:4, end-1, :] .= P.arr[1:4, end-3, :]
    P.arr[1:4, end, :] .= P.arr[1:4, end-3, :]
end

if MPI.Comm_rank(comm) == 0
    println("dt: ", dt)
end
drops::Type = T/100.0
SizeX = 16
SizeY = 16
CuP = Verona2D.CuParVector2D{Type}(P)
CUDA.@time Verona2D.HARM_HLL(
    comm,
    CuP,
    MPI_X,
    MPI_Y,
    SizeX,
    SizeY,
    dt,
    dx,
    dy,
    T,
    eos,
    drops,
    floor,
    ARGS[1],
    n_it,
    tol,
    TurnOff,
)

MPI.Finalize()
