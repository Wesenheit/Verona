@inline function kernel_Update!(U, Ubuff, dt, dx, dy, dz, Fluxes_1,Fluxes_2,Fluxes_3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    Nx, Ny, Nz = size(U,2), size(U,3), size(U,4)
    
    if 4 <= i <= Nx-3 && 4 <= j <= Ny-3 && 4 <= k <= Nz-3
         for idx in 1:5
            Ubuff[idx,i,j,k] = U[idx,i,j,k] -
                dt/dx * (Fluxes_1[idx, i+1, j  , k  ] - Fluxes_1[idx, i, j, k]) -
                dt/dy * (Fluxes_2[idx, i,   j+1, k  ] - Fluxes_2[idx, i, j, k]) -
                dt/dz * (Fluxes_3[idx, i,   j,   k+1] - Fluxes_3[idx, i, j, k])
        end
    end

    return
end
