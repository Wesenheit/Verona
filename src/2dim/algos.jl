const x = UInt8(1)
const y = UInt8(2)


@kernel inbounds = true function function_Limit(P::AbstractArray{T},floor::T) where T
    i, j = @index(Global, NTuple)
    P[1,i,j] = max(P[1,i,j],floor)
    P[2,i,j] = max(P[2,i,j],floor)
end

@kernel inbounds = true function function_Fluxes(@Const(P::AbstractArray{T}),eos::Polytrope{T},floor::T,Fglob::AbstractArray{T},dim::UInt8) where T
    i, j = @index(Global, NTuple)
    il, jl = @index(Local, NTuple)
    
    #Int32 variables take less register space
    #i = Int32(i)
    #j = Int32(j)
    #il = Int32(il)
    #jl = Int32(jl)
   
    @uniform begin
        Nx,Ny = @ndrange()
        N,M = @groupsize() #size of the local threads
    end
    ###paramters on the grid 
    # sometimes it is more beneficient to put some values in the shared memory, sometimes it is more beneficien to put them in registers
    
    PL_arr = @localmem eltype(P) (4,N, M)
    PR_arr = @localmem eltype(P) (4,N, M)

    PL = @view PL_arr[:,il,jl]
    PR = @view PR_arr[:,il,jl]
    #PL = @MVector zeros(T,4)
    #PR = @MVector zeros(T,4)
    
    FL_arr = @localmem eltype(P) (4,N, M)
    FR_arr = @localmem eltype(P) (4,N, M)
    FR = @view FR_arr[:,il,jl]
    FL = @view FL_arr[:,il,jl]
    #FR = @MVector zeros(T,4)
    #FL = @MVector zeros(T,4)
    
    #UL = @MVector zeros(T,4)
    #UR = @MVector zeros(T,4)

    UL_arr = @localmem eltype(P) (4,N, M)
    UR_arr = @localmem eltype(P) (4,N, M)
    UR = @view UR_arr[:,il,jl]
    UL = @view UL_arr[:,il,jl]

    for idx in 3:4
        PL[idx] = 0.
        PR[idx] = 0.
    end

    if i > 2 && i < Nx - 1 && j > 2 && j < Ny - 1
        for idx in 1:4
            if dim == x
                q_i = P[idx,i,j]
                q_im1 = P[idx,i-1,j]
                q_im2 = P[idx,i-2,j]
                q_ip1 = P[idx,i+1,j]
                q_ip2 = P[idx,i+2,j]
            elseif dim == y
                q_i = P[idx,i,j]
                q_im1 = P[idx,i,j-1]
                q_im2 = P[idx,i,j-2]
                q_ip1 = P[idx,i,j+1]
                q_ip2 = P[idx,i,j+2]
            end
            Q_D,Q_L = WENOZ(q_im2,q_im1,q_i,q_ip1,q_ip2)
            PL[idx] = Q_L 
        end
    end

    if i > 1 && j > 1 && j < Nx-2 && j < Ny-2
        for idx in 1:4
            if dim == x
                q_i = P[idx,i+1,j]
                q_im1 = P[idx,i,j]
                q_im2 = P[idx,i-1,j]
                q_ip1 = P[idx,i+2,j]
                q_ip2 = P[idx,i+3,j]
            elseif dim == y
                q_i = P[idx,i,j+1]
                q_im1 = P[idx,i,j]
                q_im2 = P[idx,i,j-1]
                q_ip1 = P[idx,i,j+2]
                q_ip2 = P[idx,i,j+3]
            end

            Q_D,Q_L = WENOZ(q_im2,q_im1,q_i,q_ip1,q_ip2)
            PR[idx] = Q_D
        end
    end
    
    for idx in 1:2
        PL[idx] = max(floor,PL[idx])
        PR[idx] = max(floor,PR[idx])
    end
    
    @synchronize #Only god knows why this synchronization is crucial
    
    if i > 2 && j > 2 && i < Nx-2 && j < Ny-2
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
        end
    
        lorL = sqrt(PL[3]^2 + PL[4]^2 + 1)
        lorR = sqrt(PR[3]^2 + PR[4]^2 + 1)
        if dim == x
            vL = PL[3] / lorL
            vR = PR[3] / lorR
        elseif dim == y
            vL = PL[4] / lorL
            vR = PR[4] / lorR
        end
        CL = SoundSpeed(PL[1],PL[2],eos)
        CR = SoundSpeed(PR[1],PR[2],eos)

        
        sigma_S_L = CL^2 / ( lorL^2 * (1-CL^2))
        sigma_S_R = CR^2 / ( lorR^2 * (1-CR^2))

        C_max_X = max( (vL + sqrt(sigma_S_L * (1-vL^2 + sigma_S_L)) ) / (1 + sigma_S_L), (vR + sqrt(sigma_S_R * (1-vR^2 + sigma_S_R)) ) / (1 + sigma_S_R)) # velocity composition
        C_min_X = -min( (vL - sqrt(sigma_S_L * (1-vL^2 + sigma_S_L)) ) / (1 + sigma_S_L), (vR - sqrt(sigma_S_R * (1-vR^2 + sigma_S_R)) ) / (1 + sigma_S_R)) # velocity composition
        if C_max_X < 0 
            for idx in 1:4
                Fglob[idx,i,j] =  FR[idx]
            end
        elseif C_min_X < 0 
            for idx in 1:4
                Fglob[idx,i,j] =  FL[idx] 
            end
        else
            for idx in 1:4
                Fglob[idx,i,j] = ( FR[idx] * C_min_X + FL[idx] * C_max_X - C_max_X * C_min_X * (UR[idx] - UL[idx])) / (C_max_X + C_min_X)
            end
        end
    end
end

@kernel inbounds = true function function_Update(U::AbstractArray{T},Ubuff::AbstractArray{T},dt::T,dx::T,dy::T,Fx::AbstractArray{T},Fy::AbstractArray{T}) where T
    i, j = @index(Global, NTuple)    
    Nx,Ny = @uniform @ndrange()
    
    if i > 3 && j > 3 && i < Nx-2 && j < Ny-2
        for idx in 1:4
            Ubuff[idx,i,j] = U[idx,i,j] - dt/dx * (Fx[idx,i,j] - Fx[idx,i-1,j]) - dt/dy * (Fy[idx,i,j] - Fy[idx,i,j-1])
        end
    end
end


function HARM_HLL(comm,P::VeronaArr,XMPI::Int64,YMPI::Int64,
                                    SizeX::Int64,SizeY::Int64,
                                    dt::T,dx::T,dy::T,
                                    Tmax::T,eos::EOS{T},drops::T,
                                    floor::T = 1e-7,out_dir::String = ".",kwargs...) where T

    backend = KernelAbstractions.get_backend(P.arr)
    U = VectorLike(P)
    Uhalf = VectorLike(P)
    Fx = VectorLike(P)
    Fy = VectorLike(P)

    buff_X_1 = allocate(backend,T,4,3,P.size_Y)
    buff_X_2 = allocate(backend,T,4,3,P.size_Y)
    buff_X_3 = allocate(backend,T,4,3,P.size_Y)
    buff_X_4 = allocate(backend,T,4,3,P.size_Y)
    buff_Y_1 = allocate(backend,T,4,P.size_X,3)
    buff_Y_2 = allocate(backend,T,4,P.size_X,3)
    buff_Y_3 = allocate(backend,T,4,P.size_X,3)
    buff_Y_4 = allocate(backend,T,4,P.size_X,3)
    t::T = 0

    SendBoundaryX(P,comm,buff_X_1,buff_X_2)
    SendBoundaryY(P,comm,buff_Y_1,buff_Y_2)
    WaitForBoundary(P,comm,buff_X_3,buff_X_4,buff_Y_3,buff_Y_4)

    @assert mod(P.size_X,SizeX) == 0 
    @assert mod(P.size_Y,SizeY) == 0 
    wgX = div(P.size_X,SizeX)
    wgY = div(P.size_Y,SizeY)

    Fluxes = function_Fluxes(backend, (SizeX,SizeY), (P.size_X,P.size_Y))
    Update = function_Update(backend, (SizeX,SizeY),(P.size_X,P.size_Y))
    UtoP = function_UtoP(backend, (SizeX,SizeY), (P.size_X,P.size_Y))
    PtoU = kernel_PtoU(backend, (SizeX,SizeY), (P.size_X,P.size_Y))
    Limit = function_Limit(backend, (SizeX,SizeY), (P.size_X,P.size_Y))
    
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
    
    to_save = Dict("T"=>t, "grid"=>[dx,dy])
    name = out_dir * "/dump"*string(i)*".h5"
    SaveHDF5Gather(comm,P,XMPI,YMPI,name,to_save) #save initial timestep as 0th dump

    while t < Tmax
        if length(kwargs) > 2
            fun_bound(P,t)
        end

        begin
            Fluxes(P.arr,eos,floor,Fx.arr,x)
            KernelAbstractions.synchronize(backend)
            Fluxes(P.arr,eos,floor,Fy.arr,y)
            KernelAbstractions.synchronize(backend)

            Update(U.arr,Uhalf.arr,dt/2,dx,dy,Fx.arr,Fy.arr)
            KernelAbstractions.synchronize(backend)
        end

        begin
            UtoP(Uhalf.arr,P.arr,eos,kwargs[1],kwargs[2]) #Conversion to primitive variables at the half-step
            KernelAbstractions.synchronize(backend)
            Limit(P.arr,floor)
            KernelAbstractions.synchronize(backend)
        end
        
        
        SendBoundaryX(P,comm,buff_X_1,buff_X_2)
        SendBoundaryY(P,comm,buff_Y_1,buff_Y_2)
        WaitForBoundary(P,comm,buff_X_3,buff_X_4,buff_Y_3,buff_Y_4)

        #####
        #Start of the second cycle
        #
        #Calculate Flux
        begin
            Fluxes(P.arr,eos,floor,Fx.arr,x)
            KernelAbstractions.synchronize(backend)
            Fluxes(P.arr,eos,floor,Fy.arr,y)
            KernelAbstractions.synchronize(backend)
            Update(U.arr,U.arr,dt,dx,dy,Fx.arr,Fy.arr)
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
        WaitForBoundary(P,comm,buff_X_3,buff_X_4,buff_Y_3,buff_Y_4)
      
        t += dt
        if t > thres_to_dump
            i+=1
            thres_to_dump += drops
            if MPI.Comm_rank(comm) == 0
                println(t," elapsed: ",time() - t0, " s")
                t0 = time()
            end
            to_save = Dict("T"=>t, "grid"=>[dx,dy])
            name = out_dir * "/dump"*string(i)*".h5"
            SaveHDF5Gather(comm,P,XMPI,YMPI,name,to_save)
        end
    end    
    return i
end
