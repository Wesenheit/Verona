using MPI
MPI.Init()

using Base.Threads
using CUDA
using Verona
using Random

Type = Float64

@assert MPI.has_cuda()
comm = MPI.COMM_WORLD

rank = MPI.Comm_rank(comm)

if CUDA.functional()
    CUDA.device!(rank)
    @info "Process $rank using GPU $(CUDA.device())"
end

MPI_X = 1
MPI_Y = 1
MPI_Z = 1
@assert MPI_X * MPI_Y * MPI_Z == MPI.Comm_size(comm)
comm = MPI.Cart_create(
    comm,
    (MPI_X, MPI_Y, MPI_Z),
    periodic = (false, false, false),
    reorder = true,
)

eos = Verona.EosTypes.Polytrope{Type}(4.0/3.0)
Nx = 256 - 6
Ny = 256 - 6
Nz = 256 - 6

P = Verona3D.ParVector3D{Type}(Nx, Ny, Nz)
tot_X = MPI_X * Nx + 6
tot_Y = MPI_Y * Ny + 6
tot_Z = MPI_Z * Nz + 6

idx, idy, idz = MPI.Cart_coords(comm)

#one unit -> 10^8cm

floor::Type = 1e-8
outer::Type = 1e-4
box_X::Type = 480.0
box_Y::Type = 120.0
box_Z::Type = 120.0
R_max::Type = 100.0
R_eng::Type = 5
Rho0::Type = 2.5e-2
Temp::Type = 1e-5
U0::Type = 2.5e-1

dx::Type = box_X / tot_X
dy::Type = 2*box_Y/tot_Y
dz::Type = 2*box_Z/tot_Z
Gamma0::Type = 10.0

if rank == 0
    println("dx: ", dx)
    println("dy: ", dy)
    println("dz: ", dz)
end
seed = 42

const thread_rngs = [MersenneTwister(seed + i) for i = 1:Threads.nthreads()]
start_calc = time()

Threads.@threads for num = 1:(P.size_X*P.size_Y*P.size_Z)

    cart_idx = CartesianIndices((P.size_X, P.size_Y, P.size_Z))[num]
    i, j, k = Tuple(cart_idx)
    i_g = Verona.local_to_global(i, idx, P.size_X, MPI_X)
    j_g = Verona.local_to_global(j, idy, P.size_Y, MPI_Y)
    k_g = Verona.local_to_global(k, idz, P.size_Z, MPI_Z)

    if i_g == 0 || j_g == 0 || k_g == 0
        continue
    end

    X = i_g * dx
    Y = j_g * dy - box_Y
    Z = k_g * dz - box_Z

    @fastmath begin
        R = sqrt(X^2 + Y^2 + Z^2)
        if R < R_max
            if R < R_eng
                ρ =
                    Rho0 *
                    (R_max/R_eng)^2 *
                    (1 + randn(thread_rngs[Threads.threadid()]) * 3e-2)
            else
                ρ = Rho0 * (R_max/R)^2 * (1 + randn(thread_rngs[Threads.threadid()]) * 3e-2)
            end
        else
            ρ = outer
        end

        v1 = 0.0
        v2 = 0.0
        v3 = 0.0
        u = Temp

        if R < R_eng
            ρ = ρ * 1e-2
            u = 1e-5
            r⊥ = sqrt(Y^2 + Z^2)
            prof = 1 / cosh(r⊥ / R_eng)^8

            v0 = sqrt(1 - 1 / (Gamma0^2))
            v1 = v0 * prof
            v2 = 0.0
            v3 = 0.0
        end

        ρ = max(floor, ρ)
        u = max(1e-12, u)

        v2tot = v1*v1 + v2*v2 + v3*v3
        if v2tot >= 1
            s = sqrt((1 - 1e-12) / v2tot)
            v1 *= s;
            v2 *= s;
            v3 *= s
        end

        P.arr[1, i, j, k] = ρ
        P.arr[2, i, j, k] = v1
        P.arr[3, i, j, k] = v2
        P.arr[4, i, j, k] = v3
        P.arr[5, i, j, k] = u
    end
end
end_calc = time()

Cmax::Type = 0.4
dt::Type = Cmax / (1/dx + 1/dy + 1/dz)
T::Type = box_X
n_it::Int64 = 50.0
tol::Type = 1e-6
reconstruction_method = Val(:PPM)
T_exp::Type = box_X/10

function TurnOff(P, t)
    if t > box_X/3 && t < box_X
        P.arr[2, 1:3, :, :] .*= (1-dt/T_exp)
    end
    #zero grad Y
    P.arr[:, :, 1, :] .= P.arr[:, :, 4, :]
    P.arr[:, :, 2, :] .= P.arr[:, :, 4, :]
    P.arr[:, :, 3, :] .= P.arr[:, :, 4, :]

    P.arr[:, :, end-2, :] .= P.arr[:, :, end-3, :]
    P.arr[:, :, end-1, :] .= P.arr[:, :, end-3, :]
    P.arr[:, :, end, :] .= P.arr[:, :, end-3, :]

    #zero grad Z
    P.arr[:, :, :, 1] .= P.arr[:, :, :, 4]
    P.arr[:, :, :, 2] .= P.arr[:, :, :, 4]
    P.arr[:, :, :, 3] .= P.arr[:, :, :, 4]

    P.arr[:, :, :, end-2] .= P.arr[:, :, :, end-3]
    P.arr[:, :, :, end-1] .= P.arr[:, :, :, end-3]
    P.arr[:, :, :, end] .= P.arr[:, :, :, end-3]

    #zero grad X
    P.arr[:, end-2, :, :] .= P.arr[:, end-3, :, :]
    P.arr[:, end-1, :, :] .= P.arr[:, end-3, :, :]
    P.arr[:, end, :, :] .= P.arr[:, end-3, :, :]
end
TurnOff(P, t, _idxs, _dims) = TurnOff(P, t)

if MPI.Comm_rank(comm) == 0
    println("dt: ", dt)
    println("grid setup ", end_calc-start_calc)
    println("Threads: ", nthreads())
end
drops::Type = T/1000.0
SizeX = 4
SizeY = 4
SizeZ = 4
CuP = Verona3D.CuParVector3D{Type}(P)
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
    TurnOff,
    reconstruction_method,
)

MPI.Finalize()
