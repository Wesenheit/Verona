const x = UInt8(1)
const y = UInt8(2)
const z = UInt8(3)

@kernel inbounds = true function function_Limit(
    P::AbstractArray{T},
    floor::T,
) where {T<:Real}
    i, j, k = @index(Global, NTuple)
    P[1, i, j, k] = max(P[1, i, j, k], floor)
    P[5, i, j, k] = max(P[5, i, j, k], floor)
end

@kernel inbounds = true function function_Fluxes(
    @Const(P::AbstractArray{T}),
    eos::Polytrope{T},
    floor::T,
    Fglob::AbstractArray{T},
    dim::UInt8,
) where {T<:Real}
    i, j, k = @index(Global, NTuple)
    il, jl, kl = @index(Local, NTuple)

    @uniform begin
        Nx, Ny, Nz = @ndrange()
        N, M, L = @groupsize()
    end

    PL_arr = @localmem eltype(P) (5, N, M, L)
    PR_arr = @localmem eltype(P) (5, N, M, L)

    PL = @view PL_arr[:, il, jl, kl]
    PR = @view PR_arr[:, il, jl, kl]

    FL_arr = @localmem eltype(P) (5, N, M, L)
    FR_arr = @localmem eltype(P) (5, N, M, L)
    FR = @view FR_arr[:, il, jl, kl]
    FL = @view FL_arr[:, il, jl, kl]

    UL_arr = @localmem eltype(P) (5, N, M, L)
    UR_arr = @localmem eltype(P) (5, N, M, L)
    UR = @view UR_arr[:, il, jl, kl]
    UL = @view UL_arr[:, il, jl, kl]

    eps_guard = T === Float32 ? T(1e-6) : T(1e-12)
    v2max = one(T) - eps_guard

    if i > 2 && i < Nx - 2 && j > 2 && j < Ny - 1 && k > 2 && k < Nz - 1
        failL = false
        for idx = 1:5
            if dim == x
                q_i = P[idx, i, j, k]
                q_im1 = P[idx, i-1, j, k]
                q_im2 = P[idx, i-2, j, k]
                q_ip1 = P[idx, i+1, j, k]
                q_ip2 = P[idx, i+2, j, k]
            elseif dim == y
                q_i = P[idx, i, j, k]
                q_im1 = P[idx, i, j-1, k]
                q_im2 = P[idx, i, j-2, k]
                q_ip1 = P[idx, i, j+1, k]
                q_ip2 = P[idx, i, j+2, k]
            else
                q_i = P[idx, i, j, k]
                q_im1 = P[idx, i, j, k-1]
                q_im2 = P[idx, i, j, k-2]
                q_ip1 = P[idx, i, j, k+1]
                q_ip2 = P[idx, i, j, k+2]
            end
            Q_D, Q_U = WENOZ(q_im2, q_im1, q_i, q_ip1, q_ip2)
            PL[idx] = Q_U

            qmin = min(min(q_im2, q_im1), min(q_i, min(q_ip1, q_ip2)))
            qmax = max(max(q_im2, q_im1), max(q_i, max(q_ip1, q_ip2)))
            tolQ = eps_guard * (abs(q_im2) + abs(q_im1) + abs(q_i) + abs(q_ip1) + abs(q_ip2) + one(T))
            if !(isfinite(Q_U) && Q_U >= qmin - tolQ && Q_U <= qmax + tolQ)
                failL = true
            end
        end

        v2L = PL[2]*PL[2] + PL[3]*PL[3] + PL[4]*PL[4]
        if (PL[1] < floor) || (PL[5] < floor) || !(isfinite(v2L)) || (v2L >= v2max) ||
           (!isfinite(PL[1]) || !isfinite(PL[2]) || !isfinite(PL[3]) || !isfinite(PL[4]) || !isfinite(PL[5]))
            failL = true
        end

        if failL
            failL2 = false
            for idx = 1:5
                if dim == x
                    q_i = P[idx, i, j, k]
                    q_im1 = P[idx, i-1, j, k]
                    q_im2 = P[idx, i-2, j, k]
                    q_ip1 = P[idx, i+1, j, k]
                    q_ip2 = P[idx, i+2, j, k]
                elseif dim == y
                    q_i = P[idx, i, j, k]
                    q_im1 = P[idx, i, j-1, k]
                    q_im2 = P[idx, i, j-2, k]
                    q_ip1 = P[idx, i, j+1, k]
                    q_ip2 = P[idx, i, j+2, k]
                else
                    q_i = P[idx, i, j, k]
                    q_im1 = P[idx, i, j, k-1]
                    q_im2 = P[idx, i, j, k-2]
                    q_ip1 = P[idx, i, j, k+1]
                    q_ip2 = P[idx, i, j, k+2]
                end
                Q_D, Q_U = MINMOD(q_im2, q_im1, q_i, q_ip1, q_ip2)
                PL[idx] = Q_U
                if !isfinite(Q_U)
                    failL2 = true
                end
            end

            v2L = PL[2]*PL[2] + PL[3]*PL[3] + PL[4]*PL[4]
            if (PL[1] < floor) || (PL[5] < floor) || !(isfinite(v2L)) || (v2L >= v2max) ||
               (!isfinite(PL[1]) || !isfinite(PL[2]) || !isfinite(PL[3]) || !isfinite(PL[4]) || !isfinite(PL[5]))
                failL2 = true
            end

            if failL2
                for idx = 1:5
                    PL[idx] = P[idx, i, j, k]
                end
            end
        end
    end

    if i > 1 && j > 1 && k > 1 && i < Nx - 2 && j < Ny - 2 && k < Nz - 2
        failR = false
        for idx = 1:5
            if dim == x
                q_i = P[idx, i+1, j, k]
                q_im1 = P[idx, i, j, k]
                q_im2 = P[idx, i-1, j, k]
                q_ip1 = P[idx, i+2, j, k]
                q_ip2 = P[idx, i+3, j, k]
            elseif dim == y
                q_i = P[idx, i, j+1, k]
                q_im1 = P[idx, i, j, k]
                q_im2 = P[idx, i, j-1, k]
                q_ip1 = P[idx, i, j+2, k]
                q_ip2 = P[idx, i, j+3, k]
            else
                q_i = P[idx, i, j, k+1]
                q_im1 = P[idx, i, j, k]
                q_im2 = P[idx, i, j, k-1]
                q_ip1 = P[idx, i, j, k+2]
                q_ip2 = P[idx, i, j, k+3]
            end
            Q_D, Q_L = WENOZ(q_im2, q_im1, q_i, q_ip1, q_ip2)
            PR[idx] = Q_D

            qmin = min(min(q_im2, q_im1), min(q_i, min(q_ip1, q_ip2)))
            qmax = max(max(q_im2, q_im1), max(q_i, max(q_ip1, q_ip2)))
            tolQ = eps_guard * (abs(q_im2) + abs(q_im1) + abs(q_i) + abs(q_ip1) + abs(q_ip2) + one(T))
            if !(isfinite(Q_D) && Q_D >= qmin - tolQ && Q_D <= qmax + tolQ)
                failR = true
            end
        end

        v2R = PR[2]*PR[2] + PR[3]*PR[3] + PR[4]*PR[4]
        if (PR[1] < floor) || (PR[5] < floor) || !(isfinite(v2R)) || (v2R >= v2max) ||
           (!isfinite(PR[1]) || !isfinite(PR[2]) || !isfinite(PR[3]) || !isfinite(PR[4]) || !isfinite(PR[5]))
            failR = true
        end

        if failR
            failR2 = false
            for idx = 1:5
                if dim == x
                    q_i = P[idx, i+1, j, k]
                    q_im1 = P[idx, i, j, k]
                    q_im2 = P[idx, i-1, j, k]
                    q_ip1 = P[idx, i+2, j, k]
                    q_ip2 = P[idx, i+3, j, k]
                elseif dim == y
                    q_i = P[idx, i, j+1, k]
                    q_im1 = P[idx, i, j, k]
                    q_im2 = P[idx, i, j-1, k]
                    q_ip1 = P[idx, i, j+2, k]
                    q_ip2 = P[idx, i, j+3, k]
                else
                    q_i = P[idx, i, j, k+1]
                    q_im1 = P[idx, i, j, k]
                    q_im2 = P[idx, i, j, k-1]
                    q_ip1 = P[idx, i, j, k+2]
                    q_ip2 = P[idx, i, j, k+3]
                end
                Q_D, Q_L = MINMOD(q_im2, q_im1, q_i, q_ip1, q_ip2)
                PR[idx] = Q_D
                if !isfinite(Q_D)
                    failR2 = true
                end
            end

            v2R = PR[2]*PR[2] + PR[3]*PR[3] + PR[4]*PR[4]
            if (PR[1] < floor) || (PR[5] < floor) || !(isfinite(v2R)) || (v2R >= v2max) ||
               (!isfinite(PR[1]) || !isfinite(PR[2]) || !isfinite(PR[3]) || !isfinite(PR[4]) || !isfinite(PR[5]))
                failR2 = true
            end

            if failR2
                for idx = 1:5
                    if dim == x
                        PR[idx] = P[idx, i+1, j, k]
                    elseif dim == y
                        PR[idx] = P[idx, i, j+1, k]
                    else
                        PR[idx] = P[idx, i, j, k+1]
                    end
                end
            end
        end
    end

    if i > 2 && j > 2 && k > 2 && i < Nx - 2 && j < Ny - 2 && k < Nz - 2
        for idx in (1, 5)
            PL[idx] = max(floor, PL[idx])
            PR[idx] = max(floor, PR[idx])
        end

        v2L = PL[2]*PL[2] + PL[3]*PL[3] + PL[4]*PL[4]
        if !isfinite(v2L) || v2L >= v2max
            if isfinite(v2L) && v2L > zero(T)
                sL = sqrt(v2max / v2L)
                PL[2] *= sL
                PL[3] *= sL
                PL[4] *= sL
                v2L = PL[2]*PL[2] + PL[3]*PL[3] + PL[4]*PL[4]
            else
                PL[2] = zero(T)
                PL[3] = zero(T)
                PL[4] = zero(T)
                v2L = zero(T)
            end
        end

        v2R = PR[2]*PR[2] + PR[3]*PR[3] + PR[4]*PR[4]
        if !isfinite(v2R) || v2R >= v2max
            if isfinite(v2R) && v2R > zero(T)
                sR = sqrt(v2max / v2R)
                PR[2] *= sR
                PR[3] *= sR
                PR[4] *= sR
                v2R = PR[2]*PR[2] + PR[3]*PR[3] + PR[4]*PR[4]
            else
                PR[2] = zero(T)
                PR[3] = zero(T)
                PR[4] = zero(T)
                v2R = zero(T)
            end
        end

        function_PtoU(PR, UR, eos)
        function_PtoU(PL, UL, eos)

        if dim == x
            function_PtoFx(PR, FR, eos)
            function_PtoFx(PL, FL, eos)
        elseif dim == y
            function_PtoFy(PR, FR, eos)
            function_PtoFy(PL, FL, eos)
        elseif dim == z
            function_PtoFz(PR, FR, eos)
            function_PtoFz(PL, FL, eos)
        end

        lorL = 1 / sqrt(one(T) - v2L)
        lorR = 1 / sqrt(one(T) - v2R)

        if dim == x
            vL = PL[2]
            vR = PR[2]
        elseif dim == y
            vL = PL[3]
            vR = PR[3]
        elseif dim == z
            vL = PL[4]
            vR = PR[4]
        end

        CL = (eos.gamma * PL[1] * (eos.gamma - 1) * PL[5]) / (PL[1] * (one(T) + eos.gamma * PL[5]))
        CR = (eos.gamma * PR[1] * (eos.gamma - 1) * PR[5]) / (PR[1] * (one(T) + eos.gamma * PR[5]))

        if !isfinite(CL); CL = zero(T); end
        if !isfinite(CR); CR = zero(T); end
        CL = max(zero(T), min(CL, v2max))
        CR = max(zero(T), min(CR, v2max))

        sigma_S_L = CL / (lorL*lorL * (one(T) - CL))
        sigma_S_R = CR / (lorR*lorR * (one(T) - CR))

        argL = max(zero(T), sigma_S_L * (one(T) - vL*vL + sigma_S_L))
        argR = max(zero(T), sigma_S_R * (one(T) - vR*vR + sigma_S_R))

        C_max_X = max(
            (vL + sqrt(argL)) / (one(T) + sigma_S_L),
            (vR + sqrt(argR)) / (one(T) + sigma_S_R),
        )
        C_min_X = -min(
            (vL - sqrt(argL)) / (one(T) + sigma_S_L),
            (vR - sqrt(argR)) / (one(T) + sigma_S_R),
        )

        if C_max_X < zero(T)
            for idx = 1:5
                Fglob[idx, i, j, k] = FR[idx]
            end
        elseif C_min_X < zero(T)
            for idx = 1:5
                Fglob[idx, i, j, k] = FL[idx]
            end
        else
            for idx = 1:5
                Fglob[idx, i, j, k] =
                    (
                        FR[idx] * C_min_X + FL[idx] * C_max_X -
                        C_max_X * C_min_X * (UR[idx] - UL[idx])
                    ) / (C_max_X + C_min_X)
            end
        end
    end
end

@kernel inbounds = true function function_Update(
    U::AbstractArray{T},
    Ubuff::AbstractArray{T},
    dt::T,
    dx::T,
    dy::T,
    dz::T,
    Fx::AbstractArray{T},
    Fy::AbstractArray{T},
    Fz::AbstractArray{T},
) where {T<:Real}
    i, j, k = @index(Global, NTuple)
    Nx, Ny, Nz = @uniform @ndrange()

    if i > 3 && j > 3 && k > 3 && i < Nx-2 && j < Ny-2 && k < Nz-2
        for idx = 1:5
            Ubuff[idx, i, j, k] =
                U[idx, i, j, k] - dt/dx * (Fx[idx, i, j, k] - Fx[idx, i-1, j, k]) -
                dt/dy * (Fy[idx, i, j, k] - Fy[idx, i, j-1, k]) -
                dt/dz * (Fz[idx, i, j, k] - Fz[idx, i, j, k-1])
        end
    end
end


function HARM_HLL(
    comm,
    P::VeronaArr,
    MPI_dims::Tuple{Int64,Int64,Int64},
    Size::Tuple{Int64,Int64,Int64},
    dt::T,
    dx::T,
    dy::T,
    dz::T,
    Tmax::T,
    eos::EOS{T},
    drops::T,
    floor::T = 1e-7,
    out_dir::String = ".",
    verbose::Bool = false,
    kwargs...,
) where {T}

    XMPI, YMPI, ZMPI = MPI_dims
    SizeX, SizeY, SizeZ = Size
    backend = KernelAbstractions.get_backend(P.arr)
    U = VectorLike(P)
    Uhalf = VectorLike(P)
    Fx = VectorLike(P)
    Fy = VectorLike(P)
    Fz = VectorLike(P)

    buff_X_1 = allocate(backend, T, 5, 3, P.size_Y, P.size_Z)
    buff_X_2 = allocate(backend, T, 5, 3, P.size_Y, P.size_Z)
    buff_X_3 = allocate(backend, T, 5, 3, P.size_Y, P.size_Z)
    buff_X_4 = allocate(backend, T, 5, 3, P.size_Y, P.size_Z)

    buff_Y_1 = allocate(backend, T, 5, P.size_X, 3, P.size_Z)
    buff_Y_2 = allocate(backend, T, 5, P.size_X, 3, P.size_Z)
    buff_Y_3 = allocate(backend, T, 5, P.size_X, 3, P.size_Z)
    buff_Y_4 = allocate(backend, T, 5, P.size_X, 3, P.size_Z)

    buff_Z_1 = allocate(backend, T, 5, P.size_X, P.size_Y, 3)
    buff_Z_2 = allocate(backend, T, 5, P.size_X, P.size_Y, 3)
    buff_Z_3 = allocate(backend, T, 5, P.size_X, P.size_Y, 3)
    buff_Z_4 = allocate(backend, T, 5, P.size_X, P.size_Y, 3)
    t::T = 0

    SendBoundaryX(P, comm, buff_X_1, buff_X_2)
    SendBoundaryY(P, comm, buff_Y_1, buff_Y_2)
    SendBoundaryZ(P, comm, buff_Z_1, buff_Z_2)
    WaitForBoundary(P, comm, buff_X_3, buff_X_4, buff_Y_3, buff_Y_4, buff_Z_3, buff_Z_4)

    Limit = function_Limit(backend)

    Fluxes = function_Fluxes(backend, (SizeX, SizeY, SizeZ), (P.size_X, P.size_Y, P.size_Z))
    Update = function_Update(backend, (SizeX, SizeY, SizeZ), (P.size_X, P.size_Y, P.size_Z))
    UtoP = function_UtoP(backend, (SizeX, SizeY, SizeZ), (P.size_X, P.size_Y, P.size_Z))
    PtoU = kernel_PtoU(backend, (SizeX, SizeY, SizeZ), (P.size_X, P.size_Y, P.size_Z))
    Limit = function_Limit(backend, (SizeX, SizeY, SizeZ), (P.size_X, P.size_Y, P.size_Z))

    PtoU(P.arr, U.arr, eos)
    KernelAbstractions.synchronize(backend)
    thres_to_dump::T = drops
    i::Int64 = 0
    if MPI.Comm_rank(comm) == 0
        t0 = time()
    end
    if length(kwargs) > 2
        fun_bound = kwargs[3]
    end

    to_save = Dict("T"=>t, "grid"=>[dx, dy, dz])
    name = out_dir * "/dump" * string(i) * ".h5"
    SaveHDF5Parallel(comm, P, XMPI, YMPI, ZMPI, name, to_save) #save initial timestep as 0th dump
    idx_mpi = MPI.Cart_coords(comm)
    while t < Tmax
        if length(kwargs) > 2
            fun_bound(P, t, idx_mpi, MPI_dims)
        end

        begin
            Fluxes(P.arr, eos, floor, Fx.arr, x)
            Fluxes(P.arr, eos, floor, Fy.arr, y)
            Fluxes(P.arr, eos, floor, Fz.arr, z)
            KernelAbstractions.synchronize(backend)

            Update(U.arr, Uhalf.arr, dt/2, dx, dy, dz, Fx.arr, Fy.arr, Fz.arr)
            KernelAbstractions.synchronize(backend)
        end

        begin
            UtoP(Uhalf.arr, P.arr, eos, kwargs[1], kwargs[2])
            KernelAbstractions.synchronize(backend)
            Limit(P.arr, floor)
            KernelAbstractions.synchronize(backend)
        end


        SendBoundaryX(P, comm, buff_X_1, buff_X_2)
        SendBoundaryY(P, comm, buff_Y_1, buff_Y_2)
        SendBoundaryZ(P, comm, buff_Z_1, buff_Z_2)
        WaitForBoundary(P, comm, buff_X_3, buff_X_4, buff_Y_3, buff_Y_4, buff_Z_3, buff_Z_4)

        #####
        #Start of the second cycle
        #
        #Calculate Flux
        begin
            Fluxes(P.arr, eos, floor, Fx.arr, x)
            Fluxes(P.arr, eos, floor, Fy.arr, y)
            Fluxes(P.arr, eos, floor, Fz.arr, z)
            KernelAbstractions.synchronize(backend)
            Update(U.arr, U.arr, dt, dx, dy, dz, Fx.arr, Fy.arr, Fz.arr)
            KernelAbstractions.synchronize(backend)
        end

        #sync flux on the boundaries

        begin
            UtoP(U.arr, P.arr, eos, kwargs[1], kwargs[2]) #Conversion to primitive variables at the half-step
            KernelAbstractions.synchronize(backend)
            Limit(P.arr, floor)
            KernelAbstractions.synchronize(backend)
        end

        SendBoundaryX(P, comm, buff_X_1, buff_X_2)
        SendBoundaryY(P, comm, buff_Y_1, buff_Y_2)
        SendBoundaryZ(P, comm, buff_Z_1, buff_Z_2)
        WaitForBoundary(P, comm, buff_X_3, buff_X_4, buff_Y_3, buff_Y_4, buff_Z_3, buff_Z_4)

        t += dt
        if t > thres_to_dump
            i+=1
            thres_to_dump += drops
            dims, periods, coords = MPI.Cart_get(comm)

            if MPI.Comm_rank(comm) == 0
                elapsed = time()-t0
                if verbose
                    println(
                        round(t, sigdigits = 3),
                        " elapsed: ",
                        round(elapsed, sigdigits = 3),
                        " s",
                    )
                    println(
                        "speed: ",
                        round(
                            P.size_X * P.size_Y * P.size_Z * drops/dt * 1 / elapsed *
                            10^(-6),
                            sigdigits = 3,
                        ),
                        " zones [10^6/s]",
                    )
                end
                t0 = time()
            end
            to_save = Dict("T"=>t, "grid"=>[dx, dy, dz])
            name = out_dir * "/dump" * string(i) * ".h5"
            SaveHDF5Parallel(comm, P, XMPI, YMPI, ZMPI, name, to_save)
        end
    end
    return i
end
