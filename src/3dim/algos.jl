const x = UInt8(1)
const y = UInt8(2)
const z = UInt8(3)

@kernel inbounds = true function function_Limit(P::AbstractArray{T},floor::T) where T <: Real
    i, j, k = @index(Global, NTuple)
    P[1,i,j,k] = max(P[1,i,j,k],floor)
    P[2,i,j,k] = max(P[2,i,j,k],floor)
end

@kernel inbounds = true function function_Fluxes(@Const(P::AbstractArray{T}),eos::Polytrope{T},floor::T,Fglob::AbstractArray{T},dim::UInt8) where T <: Real
    i, j, k = @index(Global, NTuple)
    il, jl,kl = @index(Local, NTuple)
   
    @uniform begin
        Nx,Ny,Nz = @ndrange()
        N,M,L = @groupsize()
    end
    #size of the local threads

    ###parameters on the grid 
    # sometimes it is more beneficiant to put some values in the shared memory, sometimes it is more beneficient to put them in registers
    
    PL_arr = @localmem eltype(P) (5,N, M, L)
    PR_arr = @localmem eltype(P) (5,N, M, L)

    PL = @view PL_arr[:,il,jl,kl]
    PR = @view PR_arr[:,il,jl,kl]
    #PL = @MVector zeros(T,4)
    #PR = @MVector zeros(T,4)
    
    FL_arr = @localmem eltype(P) (5,N, M,L)
    FR_arr = @localmem eltype(P) (5,N, M,L)
    FR = @view FR_arr[:,il,jl,kl]
    FL = @view FL_arr[:,il,jl,kl]
    #FR = @MVector zeros(T,4)
    #FL = @MVector zeros(T,4)
    
    #UL = @MVector zeros(T,4)
    #UR = @MVector zeros(T,4)

    UL_arr = @localmem eltype(P) (5,N, M, L)
    UR_arr = @localmem eltype(P) (5,N, M, L)
    UR = @view UR_arr[:,il,jl,kl]
    UL = @view UL_arr[:,il,jl,kl]


    if i > 2 && i < Nx - 2 && j > 2 && j < Ny - 1 && k > 2 && k < Nz-1
        for idx in 1:5
            if dim == x
                q_i = P[idx,i,j,k]
                q_im1 = P[idx,i-1,j,k]
                q_im2 = P[idx,i-2,j,k]
                q_ip1 = P[idx,i+1,j,k]
                q_ip2 = P[idx,i+2,j,k]
            elseif dim == y
                q_i = P[idx,i,j,k]
                q_im1 = P[idx,i,j-1,k]
                q_im2 = P[idx,i,j-2,k]
                q_ip1 = P[idx,i,j+1,k]
                q_ip2 = P[idx,i,j+2,k]
            else
                q_i = P[idx,i,j,k]
                q_im1 = P[idx,i,j,k-1]
                q_im2 = P[idx,i,j,k-2]
                q_ip1 = P[idx,i,j,k+1]
                q_ip2 = P[idx,i,j,k+2]
            end
            Q_D,Q_U = WENOZ(q_im2,q_im1,q_i,q_ip1,q_ip2)
            PL[idx] = Q_U 
        end
    end

    if i > 1 && j > 1 && k > 1 && j < Nx-2 && j < Ny-2 && k < Nz-2
        for idx in 1:5
            if dim == x
                q_i = P[idx,i+1,j,k]
                q_im1 = P[idx,i,j,k]
                q_im2 = P[idx,i-1,j,k]
                q_ip1 = P[idx,i+2,j,k]
                q_ip2 = P[idx,i+3,j,k]
            elseif dim == y
                q_i = P[idx,i,j+1,k]
                q_im1 = P[idx,i,j,k]
                q_im2 = P[idx,i,j-1,k]
                q_ip1 = P[idx,i,j+2,k]
                q_ip2 = P[idx,i,j+3,k]
            else
                q_i = P[idx,i,j,k + 1]
                q_im1 = P[idx,i,j,k]
                q_im2 = P[idx,i,j,k-1]
                q_ip1 = P[idx,i,j,k+2]
                q_ip2 = P[idx,i,j,k+3]
            end

            Q_D,Q_L = WENOZ(q_im2,q_im1,q_i,q_ip1,q_ip2)
            PR[idx] = Q_D
        end
    end
    

    if i > 2 && j > 2 && k > 2 && i < Nx-2 && j < Ny-2 && k < Nz-2
        for idx in 1:2
            PL[idx] = max(floor,PL[idx])
            PR[idx] = max(floor,PR[idx])
        end
        
        function_PtoU(PR,UR,eos)
        function_PtoU(PL,UL,eos)
        if dim == x
            function_PtoFx(PR,FR,eos)
            function_PtoFx(PL,FL,eos)
        elseif dim == y
            function_PtoFy(PR,FR,eos)
            function_PtoFy(PL,FL,eos)
        elseif dim == z
            function_PtoFz(PR,FR,eos)
            function_PtoFz(PL,FL,eos)
        end
    
        lorL = sqrt(PL[3]^2 + PL[4]^2 + PL[5]^2 + 1)
        lorR = sqrt(PR[3]^2 + PR[4]^2 + PR[5]^2 + 1)
        if dim == x
            vL = PL[3] / lorL
            vR = PR[3] / lorR
        elseif dim == y
            vL = PL[4] / lorL
            vR = PR[4] / lorR
        elseif dim == z
            vL = PL[5] / lorL
            vR = PR[5] / lorR
        end

        CL = SoundSpeed(PL[1],PL[2],eos)
        CR = SoundSpeed(PR[1],PR[2],eos)
        
        sigma_S_L = CL^2 / ( lorL^2 * (1-CL^2))
        sigma_S_R = CR^2 / ( lorR^2 * (1-CR^2))

        C_max_X = max( (vL + sqrt(sigma_S_L * (1-vL^2 + sigma_S_L)) ) / (1 + sigma_S_L), (vR + sqrt(sigma_S_R * (1-vR^2 + sigma_S_R)) ) / (1 + sigma_S_R)) # velocity composition
        C_min_X = -min( (vL - sqrt(sigma_S_L * (1-vL^2 + sigma_S_L)) ) / (1 + sigma_S_L), (vR - sqrt(sigma_S_R * (1-vR^2 + sigma_S_R)) ) / (1 + sigma_S_R)) # velocity composition
        if C_max_X < 0 
            for idx in 1:5
                Fglob[idx,i,j,k] =  FR[idx]
            end
        elseif C_min_X < 0 
            for idx in 1:5
                Fglob[idx,i,j,k] =  FL[idx] 
            end
        else
            for idx in 1:5
                Fglob[idx,i,j,k] = ( FR[idx] * C_min_X + FL[idx] * C_max_X - C_max_X * C_min_X * (UR[idx] - UL[idx])) / (C_max_X + C_min_X)
            end
        end
    end
end

@kernel inbounds = true function function_Update(U::AbstractArray{T},Ubuff::AbstractArray{T},dt::T,dx::T,dy::T,dz::T,Fx::AbstractArray{T},Fy::AbstractArray{T},Fz::AbstractArray{T}) where T <: Real
    i, j, k = @index(Global, NTuple)    
    Nx,Ny,Nz = @uniform @ndrange()
    
    if i > 3 && j > 3 && k > 3 && i < Nx-3 && j < Ny-3 && k < Nz-3
        for idx in 1:5
            Ubuff[idx,i,j,k] = U[idx,i,j,k] - dt/dx * (Fx[idx,i,j,k] - Fx[idx,i-1,j,k]) - dt/dy * (Fy[idx,i,j,k] - Fy[idx,i,j-1,k]) - dt/dz * (Fz[idx,i,j,k] - Fz[idx,i,j,k-1])
        end
    end
end


function HARM_HLL(comm,P::VeronaArr,XMPI::Int64,YMPI::Int64,ZMPI::Int64,
                                    SizeX::Int64,SizeY::Int64,SizeZ::Int64,
                                    dt::T,dx::T,dy::T,dz::T,
                                    Tmax::T,eos::EOS{T},drops::T,
                                    floor::T = 1e-7,out_dir::String = ".",kwargs...) where T

    backend = KernelAbstractions.get_backend(P.arr)
    U = VectorLike(P)
    Uhalf = VectorLike(P)
    Fx = VectorLike(P)
    Fy = VectorLike(P)
    Fz = VectorLike(P)

    buff_X_1 = allocate(backend,T,5,3,P.size_Y,P.size_Z)
    buff_X_2 = allocate(backend,T,5,3,P.size_Y,P.size_Z)
    buff_X_3 = allocate(backend,T,5,3,P.size_Y,P.size_Z)
    buff_X_4 = allocate(backend,T,5,3,P.size_Y,P.size_Z)
    
    buff_Y_1 = allocate(backend,T,5,P.size_X,3,P.size_Z)
    buff_Y_2 = allocate(backend,T,5,P.size_X,3,P.size_Z)
    buff_Y_3 = allocate(backend,T,5,P.size_X,3,P.size_Z)
    buff_Y_4 = allocate(backend,T,5,P.size_X,3,P.size_Z)
    
    buff_Z_1 = allocate(backend,T,5,P.size_X,P.size_Y,3)
    buff_Z_2 = allocate(backend,T,5,P.size_X,P.size_Y,3)
    buff_Z_3 = allocate(backend,T,5,P.size_X,P.size_Y,3)
    buff_Z_4 = allocate(backend,T,5,P.size_X,P.size_Y,3)
    t::T = 0

    SendBoundaryX(P,comm,buff_X_1,buff_X_2)
    SendBoundaryY(P,comm,buff_Y_1,buff_Y_2)
    SendBoundaryZ(P,comm,buff_Z_1,buff_Z_2)
    WaitForBoundary(P,comm,buff_X_3,buff_X_4,
                    buff_Y_3,buff_Y_4,
                    buff_Z_3,buff_Z_4)

    Limit = function_Limit(backend)

    Fluxes = function_Fluxes(backend, (SizeX,SizeY,SizeZ), (P.size_X,P.size_Y,P.size_Z))
    Update = function_Update(backend, (SizeX,SizeY,SizeZ), (P.size_X,P.size_Y,P.size_Z))
    UtoP = function_UtoP(backend, (SizeX,SizeY,SizeZ), (P.size_X,P.size_Y,P.size_Z))
    PtoU = kernel_PtoU(backend, (SizeX,SizeY,SizeZ), (P.size_X,P.size_Y,P.size_Z))
    Limit = function_Limit(backend, (SizeX,SizeY,SizeZ), (P.size_X,P.size_Y,P.size_Z))
    
    PtoU(P.arr,U.arr,eos)
    KernelAbstractions.synchronize(backend)
    thres_to_dump::T = drops
    i::Int64 = 0
    if MPI.Comm_rank(comm) == 0
        t0 = time()
    end
    if length(kwargs) > 2
        fun_bound = kwargs[3]
    end
    
    to_save = Dict("T"=>t, "grid"=>[dx,dy,dz])
    name = out_dir * "/dump"*string(i)*".h5"
    SaveHDF5Parallel(comm,P,XMPI,YMPI,ZMPI,name,to_save) #save initial timestep as 0th dump

    while t < Tmax
        if length(kwargs) > 2
            fun_bound(P,t)
        end

        begin
            Fluxes(P.arr,eos,floor,Fx.arr,x)
            Fluxes(P.arr,eos,floor,Fy.arr,y)
            Fluxes(P.arr,eos,floor,Fz.arr,z)
            KernelAbstractions.synchronize(backend)

            Update(U.arr,Uhalf.arr,dt/2,dx,dy,dz,Fx.arr,Fy.arr,Fz.arr)
            KernelAbstractions.synchronize(backend)
        end

        begin
            UtoP(Uhalf.arr,P.arr,eos,kwargs[1],kwargs[2])
            KernelAbstractions.synchronize(backend)
            Limit(P.arr,floor)
            KernelAbstractions.synchronize(backend)
        end
        
        
        SendBoundaryX(P,comm,buff_X_1,buff_X_2)
        SendBoundaryY(P,comm,buff_Y_1,buff_Y_2)
        SendBoundaryZ(P,comm,buff_Z_1,buff_Z_2)
        WaitForBoundary(P,comm,buff_X_3,buff_X_4,
                    buff_Y_3,buff_Y_4,
                    buff_Z_3,buff_Z_4)

        #####
        #Start of the second cycle
        #
        #Calculate Flux
        begin
            Fluxes(P.arr,eos,floor,Fx.arr,x)
            Fluxes(P.arr,eos,floor,Fy.arr,y)
            Fluxes(P.arr,eos,floor,Fz.arr,z)
            KernelAbstractions.synchronize(backend)
            Update(U.arr,U.arr,dt,dx,dy,dz,Fx.arr,Fy.arr,Fz.arr)
            KernelAbstractions.synchronize(backend)
        end
        
        #sync flux on the boundaries
        
        begin
            UtoP(U.arr,P.arr,eos,kwargs[1],kwargs[2]) #Conversion to primitive variables at the half-step
            KernelAbstractions.synchronize(backend)
            Limit(P.arr,floor)
            KernelAbstractions.synchronize(backend)
        end
        
        SendBoundaryX(P,comm,buff_X_1,buff_X_2)
        SendBoundaryY(P,comm,buff_Y_1,buff_Y_2)
        SendBoundaryZ(P,comm,buff_Z_1,buff_Z_2)
        WaitForBoundary(P,comm,buff_X_3,buff_X_4,
                    buff_Y_3,buff_Y_4,
                    buff_Z_3,buff_Z_4)
      
        t += dt
        if t > thres_to_dump
            i+=1
            thres_to_dump += drops
            if MPI.Comm_rank(comm) == 0
                elapsed = time()-t0
                println(round(t,sigdigits = 3)," elapsed: ",round(elapsed,sigdigits = 3), " s")
                println("speed: ",round(P.size_X * P.size_Y * P.size_Z * drops/dt * 1 / elapsed * 10^(-6),sigdigits = 3)," zones [10^6/s]")
                t0 = time()
            end
            to_save = Dict("T"=>t, "grid"=>[dx,dy,dz])
            name = out_dir * "/dump"*string(i)*".h5"
            SaveHDF5Parallel(comm,P,XMPI,YMPI,ZMPI,name,to_save)
        end
    end    
    return i
end
