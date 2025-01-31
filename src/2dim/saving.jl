
function SaveHDF5Gather(comm,P::FlowArr{T},XMPI::Int64,YMPI::Int64,name::String,to_save::Dict) where T
    size = MPI.Comm_size(comm)
    flat = vec(Array{T}( @view P.arr[:,3:end-2,3:end-2]))
    
    if MPI.Comm_rank(comm) == 0
        recvbuf = zeros(T,length(flat) *size)  # allocate buffer to recive 
    else
        recvbuf = nothing  # Non-root processes don't allocate
    end
    
    MPI.Gather!(flat, recvbuf, comm) #gather stuff
    
    if MPI.Comm_rank(comm) == 0
        global_matrix = zeros(T,4,XMPI * (P.size_X - 4),YMPI * (P.size_Y - 4))
        for p in 0:(size-1)
            px,py = MPI.Cart_coords(comm,p) #compute given coordinates
            start_x = px * (P.size_X - 4) + 1
            start_y = py * (P.size_Y - 4) + 1
            local_start = p * length(flat) + 1
            local_end = local_start + length(flat) - 1
                    
            global_matrix[:, start_x:start_x+(P.size_X - 4)-1, start_y:start_y+(P.size_Y - 4)-1] = 
                        reshape(recvbuf[local_start:local_end], 4, P.size_X-4, P.size_Y-4)
        end
        file = h5open(name,"w")
        write(file,"data",global_matrix)
        for elem in keys(to_save)
            write(file,elem,to_save[elem])
        end
        close(file)
    end
end
