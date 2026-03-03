using MPI
MPI.Init()

using Base.Threads
using CUDA
using Verona

Type = Float64

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

if CUDA.functional()
    CUDA.device!(rank)
    @info "Process $rank using GPU $(CUDA.device())"
end

MPI_X = MPI.Comm_size(comm)
MPI_Y = 1
MPI_Z = 1

comm = MPI.Cart_create(
    comm,
    (MPI_X, MPI_Y, MPI_Z),
    periodic = (false, false, false),
    reorder = true,
)

idx, idy, idz = MPI.Cart_coords(comm)

Γ::Type = 4.0 / 3.0
eos = Verona.EosTypes.Polytrope{Type}(Γ)

ρL::Type = 1.0
pL::Type = 1
vL::Type = 0.9

ρR::Type = 1.0
pR::Type = 10
vR::Type = 0.0

uL::Type = pL / (ρL * (Γ - 1))
uR::Type = pR / (ρR * (Γ - 1))

tot_X_target = 16*512
tot_Y_target = 8
tot_Z_target = 8

Nx = (tot_X_target - 6) ÷ MPI_X
Ny = tot_Y_target - 6
Nz = tot_Z_target - 6

P = Verona3D.ParVector3D{Type}(Nx, Ny, Nz)

tot_X = MPI_X * Nx + 6
tot_Y = MPI_Y * Ny + 6
tot_Z = MPI_Z * Nz + 6

box_X::Type = 1.0
box_Y::Type = 0.5
box_Z::Type = 0.5
x0::Type = 0.5

dx::Type = box_X / tot_X
dy::Type = 2 * box_Y / tot_Y
dz::Type = 2 * box_Z / tot_Z

floorρ::Type = 1e-12
flooru::Type = 1e-12

Threads.@threads for num = 1:(P.size_X*P.size_Y*P.size_Z)
    cart_idx = CartesianIndices((P.size_X, P.size_Y, P.size_Z))[num]
    i, j, k = Tuple(cart_idx)

    i_g = Verona.local_to_global(i, idx, P.size_X, MPI_X)
    j_g = Verona.local_to_global(j, idy, P.size_Y, MPI_Y)
    k_g = Verona.local_to_global(k, idz, P.size_Z, MPI_Z)

    if i_g == 0 || j_g == 0 || k_g == 0
        continue
    end

    x = i_g * dx

    ρ::Type = (x < x0) ? ρL : ρR
    v1::Type = (x < x0) ? vL : vR
    u::Type = (x < x0) ? uL : uR

    ρ = max(floorρ, ρ)
    u = max(flooru, u)

    @inbounds begin
        P.arr[1, i, j, k] = ρ
        P.arr[2, i, j, k] = v1
        P.arr[3, i, j, k] = 0.0
        P.arr[4, i, j, k] = 0.0
        P.arr[5, i, j, k] = u
    end
end

Cmax::Type = 0.4
dt::Type = Cmax / (1 / dx + 1 / dy + 1 / dz)

function BC_compressed_1D!(P, t, ids_dim, tot_dim)
    if ids_dim[2] == 0
        P.arr[:, :, 1, :] .= P.arr[:, :, 4, :]
        P.arr[:, :, 2, :] .= P.arr[:, :, 4, :]
        P.arr[:, :, 3, :] .= P.arr[:, :, 4, :]
    end

    if ids_dim[2] == tot_dim[2] - 1
        P.arr[:, :, end-2, :] .= P.arr[:, :, end-3, :]
        P.arr[:, :, end-1, :] .= P.arr[:, :, end-3, :]
        P.arr[:, :, end, :] .= P.arr[:, :, end-3, :]
    end

    if ids_dim[3] == 0
        P.arr[:, :, :, 1] .= P.arr[:, :, :, 4]
        P.arr[:, :, :, 2] .= P.arr[:, :, :, 4]
        P.arr[:, :, :, 3] .= P.arr[:, :, :, 4]
    end

    if ids_dim[3] == tot_dim[3] - 1
        P.arr[:, :, :, end-2] .= P.arr[:, :, :, end-3]
        P.arr[:, :, :, end-1] .= P.arr[:, :, :, end-3]
        P.arr[:, :, :, end] .= P.arr[:, :, :, end-3]
    end

    if ids_dim[1] == tot_dim[1] - 1
        P.arr[:, end-2, :, :] .= P.arr[:, end-3, :, :]
        P.arr[:, end-1, :, :] .= P.arr[:, end-3, :, :]
        P.arr[:, end, :, :] .= P.arr[:, end-3, :, :]
    end

    if ids_dim[1] == 0
        P.arr[:, 1, :, :] .= P.arr[:, 4, :, :]
        P.arr[:, 2, :, :] .= P.arr[:, 4, :, :]
        P.arr[:, 3, :, :] .= P.arr[:, 4, :, :]
    end
end

T_end::Type = 0.4
n_it::Int64 = 200
tol::Type = 1e-8
drops::Type = T_end/100

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
    T_end,
    eos,
    drops,
    floorρ,
    ARGS[1],
    true,
    n_it,
    tol,
    BC_compressed_1D!,
)

MPI.Finalize()
