@inline function EMF_X_kernel!(EMF_X,FluxCT_1,FluxCT_2,FluxCT_3)
	i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
	k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
		Nx, Ny, Nz = size(FluxCT_1,2), size(FluxCT_1,3), size(FluxCT_1,4)
		Nx = Nx - 1
		if 4 <= i <= Nx-3 && 4 <= j <= Ny-2 && 4 <= k <= Nz-2
			EMF_X[i,j,k] = 0.25*(FluxCT_3[2,i,j,k] + FluxCT_3[2,i,j-1,k] - FluxCT_2[3,i,j,k-1] - FluxCT_2[3,i,j,k])
		end
	return
end

@inline function EMF_Y_kernel!(EMF_Y,FluxCT_1,FluxCT_2,FluxCT_3)
	i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
	k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
		Nx, Ny, Nz = size(FluxCT_2,2), size(FluxCT_2,3), size(FluxCT_2,4)
		Ny = Ny - 1
		if 4 <= i <= Nx-2 && 4 <= j <= Ny-3 && 4 <= k <= Nz-2
			EMF_Y[i,j,k] = 0.25*(FluxCT_1[3,i,j,k] + FluxCT_1[3,i,j,k-1] - FluxCT_3[1,i,j,k] - FluxCT_3[1,i-1,j,k])
		end
	return
end

@inline function EMF_Z_kernel!(EMF_Z,FluxCT_1,FluxCT_2,FluxCT_3)
	i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
	k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
		Nx, Ny, Nz = size(FluxCT_3,2), size(FluxCT_3,3), size(FluxCT_3,4)
		Nz = Nz - 1
		if 4 <= i <= Nx-2 && 4 <= j <= Ny-2 && 4 <= k <= Nz-3
        		EMF_Z[i,j,k] = 0.25*(FluxCT_2[1,i,j,k] + FluxCT_2[1,i-1,j,k] - FluxCT_1[2,i,j,k] - FluxCT_1[2,i,j-1,k])
		end
	return
end
