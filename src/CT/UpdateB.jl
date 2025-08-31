@inline function kernel_UpdateB!(B1, B2, B3, B1_buffer, B2_buffer, B3_buffer, EMF1, EMF2, EMF3, dt, dx, dy, dz)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    Nx, Ny, Nz = size(B3,1), size(B3,2), size(B3,3) -1
    
    if 4 <= i <= Nx-2 && 4 <= j <= Ny-3 && 4 <= k <= Nz-3
        B1_buffer[i,j,k] = B1[i,j,k] - dt/dy * (EMF3[i,j+1,k]-EMF3[i,j,k]) + dt/dz * (EMF2[i,j,k+1]- EMF2[i,j,k])
    end
	
    if 4 <= i <= Nx-3 && 4 <= j <= Ny-2 && 4 <= k <= Nz-3	
	B2_buffer[i,j,k] = B2[i,j,k] + dt/dx * (EMF3[i+1,j,k]-EMF3[i,j,k]) - dt/dz * (EMF1[i,j,k+1]- EMF1[i,j,k])
    end  
    
    if 4 <= i <= Nx-3 && 4 <= j <= Ny-3 && 4 <= k <= Nz-2        
        B3_buffer[i,j,k] = B3[i,j,k] - dt/dx * (EMF2[i+1,j,k]-EMF2[i,j,k]) + dt/dy * (EMF1[i,j+1,k]- EMF1[i,j,k])
    end
    
    return
end

