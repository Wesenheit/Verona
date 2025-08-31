@inline function open_boundaries_all!(P::AbstractArray{T}) where T<:Real
    Nx, Ny, Nz = size(P,2), size(P,3), size(P,4)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if 1 <= i <= Nx && 1 <= j <= Ny && 1 <= k <= Nz
        for v in 1:5
            if i <= 3
                P[v, i, j, k] = P[v, 4, j, k]
            elseif i >= Nx - 2
                P[v, i, j, k] = P[v, Nx-3, j, k]
            end
        end
    end

    if 1 <= i <= Nx && 1 <= j <= Ny && 1 <= k <= Nz
        for v in 1:5
            if j <= 3
                P[v, i, j, k] = P[v, i, 4, k]
            elseif j >= Ny - 2
                P[v, i, j, k] = P[v, i, Ny-3, k]
            end
        end
    end

    if 1 <= i <= Nx && 1 <= j <= Ny && 1 <= k <= Nz
        for v in 1:5
            if k <= 3
                P[v, i, j, k] = P[v, i, j, 4]
            elseif k >= Nz - 2
                P[v, i, j, k] = P[v, i, j, Nz-3]
            end
        end
    end

    return
end



@inline function open_boundaries_all_B1!(B1::AbstractArray{T}, B2::AbstractArray{T}, B3::AbstractArray{T}) where T<:Real
    Nx = size(B1,1) -1
    Ny, Nz = size(B1,2), size(B1,3)

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if 1 <= j <= Ny && 1 <= k <= Nz
        B1[1,    j, k] = B1[4,    j, k]
        B1[2,    j, k] = B1[4,    j, k]
        B1[3,    j, k] = B1[4,    j, k]
        B1[Nx+1, j, k] = B1[Nx-2, j, k]
        B1[Nx,   j, k] = B1[Nx-2, j, k]
        B1[Nx-1, j, k] = B1[Nx-2, j, k]
    end
    if 1 <= i <= Nx+1 && 1 <= k <= Nz
        B1[i, 1,    k] = B1[i, 4,    k]
        B1[i, 2,    k] = B1[i, 4,    k]
        B1[i, 3,    k] = B1[i, 4,    k]
        B1[i, Ny-2, k] = B1[i, Ny-3, k]
        B1[i, Ny-1, k] = B1[i, Ny-3, k]
        B1[i, Ny,   k] = B1[i, Ny-3, k]
    end
    if 1 <= i <= Nx+1 && 1 <= j <= Ny
        B1[i, j, 1]    = B1[i, j, 4]
        B1[i, j, 2]    = B1[i, j, 4]
        B1[i, j, 3]    = B1[i, j, 4]
        B1[i, j, Nz-2] = B1[i, j, Nz-3]
        B1[i, j, Nz-1] = B1[i, j, Nz-3]
        B1[i, j, Nz]   = B1[i, j, Nz-3]
    end
    return
end

@inline function open_boundaries_all_B2!(B1::AbstractArray{T}, B2::AbstractArray{T}, B3::AbstractArray{T}) where T<:Real
    Nx = size(B2, 1)
    Ny = size(B2, 2) - 1
    Nz = size(B2, 3)

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if 1 <= j <= Ny+1 && 1 <= k <= Nz
        B2[1,    j, k] = B2[4,    j, k]
        B2[2,    j, k] = B2[4,    j, k]
        B2[3,    j, k] = B2[4,    j, k]
        B2[Nx,   j, k] = B2[Nx-3, j, k]
        B2[Nx-1, j, k] = B2[Nx-3, j, k]
        B2[Nx-2, j, k] = B2[Nx-3, j, k]
    end

    if 1 <= i <= Nx && 1 <= k <= Nz
        B2[i, 1,     k] = B2[i, 4,     k]
        B2[i, 2,     k] = B2[i, 4,     k]
        B2[i, 3,     k] = B2[i, 4,     k]
        B2[i, Ny+1,  k] = B2[i, Ny-2,  k]
        B2[i, Ny,    k] = B2[i, Ny-2,  k]
        B2[i, Ny-1,  k] = B2[i, Ny-2,  k]
    end

    if 1 <= i <= Nx && 1 <= j <= Ny+1
        B2[i, j, 1]    = B2[i, j, 4]
        B2[i, j, 2]    = B2[i, j, 4]
        B2[i, j, 3]    = B2[i, j, 4]
        B2[i, j, Nz]   = B2[i, j, Nz-3]
        B2[i, j, Nz-1] = B2[i, j, Nz-3]
        B2[i, j, Nz-2] = B2[i, j, Nz-3]
    end

    return
end

@inline function open_boundaries_all_B3!(B1::AbstractArray{T}, B2::AbstractArray{T}, B3::AbstractArray{T}) where T<:Real
    Nx = size(B3, 1)
    Ny = size(B3, 2)
    Nz = size(B3, 3) - 1

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if 1 <= j <= Ny && 1 <= k <= Nz+1
        B3[1,    j, k] = B3[4,    j, k]
        B3[2,    j, k] = B3[4,    j, k]
        B3[3,    j, k] = B3[4,    j, k]
        B3[Nx,   j, k] = B3[Nx-3, j, k]
        B3[Nx-1, j, k] = B3[Nx-3, j, k]
        B3[Nx-2, j, k] = B3[Nx-3, j, k]
    end

    if 1 <= i <= Nx && 1 <= k <= Nz+1
        B3[i, 1,    k] = B3[i, 4,    k]
        B3[i, 2,    k] = B3[i, 4,    k]
        B3[i, 3,    k] = B3[i, 4,    k]
        B3[i, Ny,   k] = B3[i, Ny-3, k]
        B3[i, Ny-1, k] = B3[i, Ny-3, k]
        B3[i, Ny-2, k] = B3[i, Ny-3, k]
    end

    if 1 <= i <= Nx && 1 <= j <= Ny
        B3[i, j, 1]     = B3[i, j, 4]
        B3[i, j, 2]     = B3[i, j, 4]
        B3[i, j, 3]     = B3[i, j, 4]
        B3[i, j, Nz+1]  = B3[i, j, Nz-2]
        B3[i, j, Nz]    = B3[i, j, Nz-2]
        B3[i, j, Nz-1]  = B3[i, j, Nz-2]
    end

    return
end



