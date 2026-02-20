using CUDA

function SendBoundaryX(U::CuParVector3D{T},comm,buff_X_1::CuArray{T},buff_X_2::CuArray{T}) where T<:Real
    CUDA.@sync begin
        CUDA.copyto!(buff_X_1, CUDA.@view(U.arr[:, end-5:end-3, :,:]))
        CUDA.copyto!(buff_X_2, CUDA.@view(U.arr[:, 4:6, :,:]))
    end
    left,right = MPI.Cart_shift(comm,0,1)
    
    MPI.Isend(buff_X_1,right,0,comm)
    MPI.Isend(buff_X_2,left,1,comm)
end

function SendBoundaryY(U::CuParVector3D{T},comm,buff_Y_1::CuArray{T},buff_Y_2::CuArray{T}) where T<:Real
    CUDA.@sync begin
        CUDA.copyto!(buff_Y_1, CUDA.@view(U.arr[:, :,end-5:end-3,:]))
        CUDA.copyto!(buff_Y_2, CUDA.@view(U.arr[:, :, 4:6,:]))
    end

    left,right = MPI.Cart_shift(comm,1,1)

    MPI.Isend(buff_Y_1,right,0,comm)                #,buff_Y_2,rank_dest_up,0,comm)
    MPI.Isend(buff_Y_2,left,1,comm)              #,buff_Y_2,rank_dest_down,1,comm)
end

function SendBoundaryZ(U::CuParVector3D{T},comm,buff_Z_1::CuArray{T},buff_Z_2::CuArray{T}) where T<:Real
    CUDA.@sync begin
        CUDA.copyto!(buff_Z_1, CUDA.@view(U.arr[:, :,:,end-5:end-3]))
        CUDA.copyto!(buff_Z_2, CUDA.@view(U.arr[:, :,:, 4:6]))
    end

    left,right = MPI.Cart_shift(comm,2,1)

    MPI.Isend(buff_Z_1,right,0,comm)                
    MPI.Isend(buff_Z_2,left,1,comm)              
end
function WaitForBoundary(U::CuParVector3D{T},comm,
                        buff_X_1::CuArray{T},buff_X_2::CuArray{T},
                        buff_Y_1::CuArray{T},buff_Y_2::CuArray{T},
                        buff_Z_1::CuArray{T},buff_Z_2::CuArray{T}) where T<:Real
    
    leftX,rightX = MPI.Cart_shift(comm,0,1)
    
    leftY,rightY = MPI.Cart_shift(comm,1,1)
    
    leftZ,rightZ = MPI.Cart_shift(comm,2,1)
    
    requests = MPI.Request[]
    
    # X direction receives
    if leftX != MPI.PROC_NULL  # Receive from left neighbor
        r1 = MPI.Irecv!(buff_X_1, leftX, 0, comm)
        push!(requests, r1)
    else
        CUDA.@sync CUDA.copyto!(buff_X_1, CUDA.@view(U.arr[:, 1:3,:,:]))
    end
    
    if rightX != MPI.PROC_NULL  # Receive from right neighbor
        r2 = MPI.Irecv!(buff_X_2, rightX, 1, comm)
        push!(requests, r2)
    else
        CUDA.@sync CUDA.copyto!(buff_X_2, CUDA.@view(U.arr[:, end-2:end,:,:]))
    end
    
    # Y direction receives  
    if leftY != MPI.PROC_NULL  # Receive from down neighbor
        r3 = MPI.Irecv!(buff_Y_1, leftY, 0, comm)
        push!(requests, r3)
    else
        # Handle bottom boundary condition
        CUDA.@sync CUDA.copyto!(buff_Y_1, CUDA.@view(U.arr[:, :,1:3,:]))
    end
    
    if rightY != MPI.PROC_NULL  # Receive from up neighbor
        r4 = MPI.Irecv!(buff_Y_2, rightY, 1, comm)
        push!(requests, r4)
    else
        CUDA.@sync CUDA.copyto!(buff_Y_2, CUDA.@view(U.arr[:, :,end-2:end,:]))
    end
    
    if leftZ != MPI.PROC_NULL  # Receive from back neighbor
        r5 = MPI.Irecv!(buff_Z_1, leftZ, 0, comm)
        push!(requests, r5)
    else
        CUDA.@sync CUDA.copyto!(buff_Z_1, CUDA.@view(U.arr[:, :,:,1:3]))
    end
    
    if rightZ != MPI.PROC_NULL  # Receive from forward neighbor
        r6 = MPI.Irecv!(buff_Z_2, rightZ, 1, comm)
        push!(requests, r6)
    else
        CUDA.@sync CUDA.copyto!(buff_Z_2, CUDA.@view(U.arr[:, :,:,end-2:end]))
    end
    
    if !isempty(requests)
        MPI.Waitall(requests)
    end
    
    CUDA.@sync begin
        if leftX != MPI.PROC_NULL
            CUDA.copyto!(CUDA.@view(U.arr[:, 1:3,:,:]), buff_X_1)
        end
        if rightX != MPI.PROC_NULL
            CUDA.copyto!(CUDA.@view(U.arr[:, end-2:end,:,:]), buff_X_2)
        end
        
        if leftY != MPI.PROC_NULL
            CUDA.copyto!(CUDA.@view(U.arr[:, :,1:3,:]), buff_Y_1)
        end
        if rightY != MPI.PROC_NULL
            CUDA.copyto!(CUDA.@view(U.arr[:, :,end-2:end,:]), buff_Y_2)
        end
        
        if leftZ != MPI.PROC_NULL
            CUDA.copyto!(CUDA.@view(U.arr[:, :,:,1:3]), buff_Z_1)
        end
        if rightZ != MPI.PROC_NULL
            CUDA.copyto!(CUDA.@view(U.arr[:, :,:,end-2:end]), buff_Z_2)
        end
    end
end
