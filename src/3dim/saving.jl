function SaveHDF5Gather(comm,P::VeronaArr{T},XMPI::Int64,YMPI::Int64,ZMPI::Int64,name::String,to_save::Dict) where T <:Real
    size = MPI.Comm_size(comm)
    flat = vec(Array{T}( @view P.arr[:,4:end-3,4:end-3,4:end-3]))
    
    if MPI.Comm_rank(comm) == 0
        recvbuf = zeros(T,length(flat) *size)  # allocate buffer to recive  
    else
        recvbuf = nothing  # Non-root processes don't allocate
    end
    
    MPI.Gather!(flat, recvbuf, comm,root = 0) #gather stuff
    
    if MPI.Comm_rank(comm) == 0
        global_matrix = zeros(T,5,XMPI * (P.size_X - 6),YMPI * (P.size_Y - 6),ZMPI*(P.size_Z - 6))
        for p in 0:(size-1)
            px,py,pz = MPI.Cart_coords(comm,p) #compute given coordinates
            start_x = px * (P.size_X - 6) + 1
            start_y = py * (P.size_Y - 6) + 1
            start_z = pz * (P.size_Z - 6) + 1
            local_start = p * length(flat) + 1
            local_end = local_start + length(flat) - 1
                    
            global_matrix[:, start_x:start_x+(P.size_X - 6)-1, start_y:start_y+(P.size_Y - 6)-1,start_z:start_z + (P.size_Z - 6)-1] = 
                        reshape(recvbuf[local_start:local_end], 5, P.size_X-6, P.size_Y-6, P.size_Z-6)
        end
        file = h5open(name,"w")
        write(file,"data",global_matrix)
        for elem in keys(to_save)
            write(file,elem,to_save[elem])
        end
        close(file)
        
        if any(isnan.(global_matrix))
            throw("Nan in matrix")
        end
    end
end
function SaveHDF5Parallel(comm, P::VeronaArr{T}, XMPI::Int, YMPI::Int, ZMPI::Int, name::String, to_save::Dict) where {T <: Real}
    rank = MPI.Comm_rank(comm)
    local_data = @view P.arr[:, 4:end-3, 4:end-3, 4:end-3]
    local_org = Array(local_data)
    
    if any(isnan.(local_data))
        throw("Nan in matrix")
    end
    
    global_size_X = XMPI * (P.size_X - 6)
    global_size_Y = YMPI * (P.size_Y - 6)
    global_size_Z = ZMPI * (P.size_Z - 6)
    
    px, py, pz = MPI.Cart_coords(comm)  # compute given coordinates
    offset_x = px * (P.size_X - 6)
    offset_y = py * (P.size_Y - 6)
    offset_z = pz * (P.size_Z - 6)
    
    local_size = size(local_data)
    
    # Create file access property list for MPI-IO
    fapl = HDF5.h5p_create(HDF5.H5P_FILE_ACCESS)
    HDF5.h5p_set_fapl_mpio(fapl, comm, MPI.INFO_NULL)
    
    # Create/open file
    fid = HDF5.h5f_create(name, HDF5.H5F_ACC_TRUNC, HDF5.H5P_DEFAULT, fapl)
    
    # Define global dimensions
    global_dims = Vector{HDF5.hsize_t}(reverse([5, global_size_X, global_size_Y, global_size_Z]))
    
    # Create filespace
    filespace = HDF5.h5s_create_simple(4, global_dims, global_dims)
    
    # Create dataset
    dset = HDF5.h5d_create(fid, "data", HDF5.H5T_NATIVE_DOUBLE, filespace,
                           HDF5.H5P_DEFAULT, HDF5.H5P_DEFAULT, HDF5.H5P_DEFAULT)

    # Define hyperslab for this process
    offset = Vector{HDF5.hsize_t}(reverse([0, offset_x, offset_y, offset_z]))
    block = Vector{HDF5.hsize_t}(reverse([5, P.size_X-6, P.size_Y-6, P.size_Z-6]))
    count = Vector{HDF5.hsize_t}([1, 1, 1, 1])
    stride = Vector{HDF5.hsize_t}([1, 1, 1, 1])
    
    # Select hyperslab in filespace
    HDF5.h5s_select_hyperslab(filespace, HDF5.H5S_SELECT_SET, offset, stride, count, block)
    
    # Create memory space matching local data dimensions
    local_dims = reverse(Vector{HDF5.hsize_t}(collect(size(local_org))))
    memspace = HDF5.h5s_create_simple(length(local_dims), local_dims, local_dims)
    
    # Create data transfer property list for collective I/O
    xfer_plist = HDF5.h5p_create(HDF5.H5P_DATASET_XFER)
    HDF5.h5p_set_dxpl_mpio(xfer_plist, HDF5.H5FD_MPIO_COLLECTIVE)
    
    # Write data
    HDF5.h5d_write(dset, HDF5.H5T_NATIVE_DOUBLE, memspace, filespace, xfer_plist, local_org)
    
    # Close spaces and property lists
    HDF5.h5s_close(memspace)
    HDF5.h5s_close(filespace)
    HDF5.h5p_close(xfer_plist)
    HDF5.h5d_close(dset)
    
    HDF5.h5f_close(fid)
    HDF5.h5p_close(fapl)
    MPI.Barrier(comm)
    if rank == 0
        file = h5open(name,"r+")
        for elem in keys(to_save)
            write(file,elem,to_save[elem])
        end
        close(file)
    end
end

