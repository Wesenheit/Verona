using CUDA

function SendBoundaryX(
    U::CuParVector2D{T},
    comm,
    buff_X_1::CuArray{T},
    buff_X_2::CuArray{T},
) where {T}
    CUDA.@sync begin
        CUDA.copyto!(buff_X_1, CUDA.@view(U.arr[:, (end-5):(end-3), :]))
        CUDA.copyto!(buff_X_2, CUDA.@view(U.arr[:, 4:6, :]))
    end
    rank_source_right, rank_dest_right = MPI.Cart_shift(comm, 0, 1)
    rank_source_left, rank_dest_left = MPI.Cart_shift(comm, 0, -1)

    MPI.Isend(buff_X_1, rank_source_right, 0, comm)
    MPI.Isend(buff_X_2, rank_source_left, 1, comm)
end

function SendBoundaryY(
    U::CuParVector2D{T},
    comm,
    buff_Y_1::CuArray{T},
    buff_Y_2::CuArray{T},
) where {T}
    CUDA.@sync begin
        CUDA.copyto!(buff_Y_1, CUDA.@view(U.arr[:, :, (end-5):(end-3)]))
        CUDA.copyto!(buff_Y_2, CUDA.@view(U.arr[:, :, 4:6]))
    end

    rank_source_up, rank_dest_up = MPI.Cart_shift(comm, 1, 1)
    rank_source_down, rank_dest_down = MPI.Cart_shift(comm, 1, -1)

    MPI.Isend(buff_Y_1, rank_source_up, 0, comm)                #,buff_Y_2,rank_dest_up,0,comm)
    MPI.Isend(buff_Y_2, rank_source_down, 1, comm)              #,buff_Y_2,rank_dest_down,1,comm)
end

function WaitForBoundary(
    U::CuParVector2D{T},
    comm,
    buff_X_1::CuArray{T},
    buff_X_2::AbstractArray{T},
    buff_Y_1::CuArray{T},
    buff_Y_2::AbstractArray{T},
) where {T}


    rank_source_right, rank_dest_right = MPI.Cart_shift(comm, 0, 1)
    rank_source_left, rank_dest_left = MPI.Cart_shift(comm, 0, -1)
    rank_source_up, rank_dest_up = MPI.Cart_shift(comm, 1, 1)
    rank_source_down, rank_dest_down = MPI.Cart_shift(comm, 1, -1)

    CUDA.@sync begin
        CUDA.copyto!(buff_Y_1, CUDA.@view(U.arr[:, :, 1:3]))
        CUDA.copyto!(buff_Y_2, CUDA.@view(U.arr[:, :, (end-2):end]))
        CUDA.copyto!(buff_X_1, CUDA.@view(U.arr[:, 1:3, :]))
        CUDA.copyto!(buff_X_2, CUDA.@view(U.arr[:, (end-2):end, :]))
    end

    r1 = MPI.Irecv!(buff_X_1, rank_dest_right, 0, comm)
    r2 = MPI.Irecv!(buff_X_2, rank_dest_left, 1, comm)
    r3 = MPI.Irecv!(buff_Y_1, rank_dest_up, 0, comm)
    r4 = MPI.Irecv!(buff_Y_2, rank_dest_down, 1, comm)
    MPI.Waitall([r1, r2, r3, r4])

    CUDA.@sync begin
        CUDA.copyto!(CUDA.@view(U.arr[:, :, 1:3]), buff_Y_1)
        CUDA.copyto!(CUDA.@view(U.arr[:, :, (end-2):end]), buff_Y_2)
        CUDA.copyto!(CUDA.@view(U.arr[:, 1:3, :]), buff_X_1)
        CUDA.copyto!(CUDA.@view(U.arr[:, (end-2):end, :]), buff_X_2)
    end
end
