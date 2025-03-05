using LinearAlgebra
using StaticArrays
using Base.Threads
using Base.Iterators
using KernelAbstractions
using CUDA
using MPI
using HDF5

# Used scheme
# U - conserved variables
# U1 = rho ut - mass conservation
# U2 = T^t_t - energy conservation
# U3 = T^t_x - momentum conservation x
# U4 = T^t_y - momentum conservation y
# U5 = T^t_z - momentum conservation z
 
# P - primitive variables
# P1 = rho - density
# P2 = u - energy density
# P3 = ux four-velocity in x
# P4 = uy four-velocity in y
# P5 = uz four-velocity in z

function local_to_global(i,p,Size,MPI)
    if p == 0
        return i
    elseif p > 0 && p < MPI-1 && (p < 4 || px > Size-3) 
        return 0
    else
        return i + p * (Size - 6)
    end
end


abstract type VeronaArr{T} end


mutable struct ParVector3D{T <:Real} <: VeronaArr{T}
    # Parameter Vector
    arr::Array{T,4}
    size_X::Int64
    size_Y::Int64
    size_Z::Int64
    function ParVector3D{T}(Nx,Ny,Nz) where {T}
        arr = zeros(T, 5, Nx + 6, Ny + 6, Nz + 6)
        new(arr,Nx + 6 , Ny + 6, Nz + 6)
    end
    function ParVector3D{T}(arr::VeronaArr{T}) where {T}
        new(Array{T}(arr.arr), arr.size_X, arr.size_Y, arr.size_Z)
    end
end

mutable struct CuParVector3D{T <:Real} <: VeronaArr{T}
    # Parameter Vector
    arr::CuArray{T}
    size_X::Int64
    size_Y::Int64
    size_Z::Int64
    function CuParVector3D{T}(arr::VeronaArr{T}) where T <:Real
        new(CuArray{T}(arr.arr), arr.size_X, arr.size_Y, arr.size_Z)
    end

    function CuParVector3D{T}(Nx::Int64, Ny::Int64, Nz::Int64) where T <:Real
        new(CuArray{T}(zeros(T, 5, Nx + 6, Ny + 6, Nz + 6)), Nx + 6, Ny + 6, Nz + 6)
    end
end

function VectorLike(X::VeronaArr{T}) where T
    if typeof(X.arr) <: CuArray
        return CuParVector3D{T}(X.size_X - 6, X.size_Y - 6, X.size_Z - 6)
    else
        return ParVector3D{T}(X.size_X - 6, X.size_Y - 6, X.size_Z - 6)
    end
end

@kernel inbounds = true function kernel_PtoU(@Const(P::AbstractArray{T}), U::AbstractArray{T},eos::Polytrope{T}) where T<:Real
    i, j, k = @index(Global, NTuple)    
    begin
        gam = sqrt(P[3,i,j,k]^2 + P[4,i,j,k]^2 + P[5,i,j,k]^2 + 1)  #gam = u⁰
        w = eos.gamma * P[2,i,j,k] + P[1,i,j,k]                     # ρ + u + p = ρ + eos.gamma*u
        U[1,i,j,k] = gam * P[1,i,j,k]
        U[2,i,j,k] = (eos.gamma-1) * P[2,i,j,k] - gam^2 * w 
        U[3,i,j,k] = P[3,i,j,k] * gam * w
        U[4,i,j,k] = P[4,i,j,k] * gam * w
        U[5,i,j,k] = P[5,i,j,k] * gam * w        
    end
end


@inline function function_PtoU(P::AbstractVector{T}, U::AbstractVector{T},eos::Polytrope{T}) where T<:Real
    gam = sqrt(P[3]^2 + P[4]^2 +P[5]^2 + 1) #gam = u⁰
    w = eos.gamma * P[2] + P[1] 
    U[1] = gam * P[1]
    U[2] = (eos.gamma-1) * P[2] - gam^2 * w
    U[3] = P[3] * gam * w
    U[4] = P[4] * gam * w
    U[5] = P[5] * gam * w
end


@inline function function_PtoFx(P::AbstractVector{T}, Fx::AbstractVector{T},eos::Polytrope{T}) where T<:Real
    gam = sqrt(P[3]^2 + P[4]^2 + P[5]^2 + 1)
    w = eos.gamma * P[2] + P[1] 
    Fx[1] = P[1] * P[3]
    Fx[2] = - w *P[3] * gam
    Fx[3] = P[3]^2 * w + (eos.gamma - 1) * P[2]
    Fx[4] = P[3] * P[4] * w 
    Fx[5] = P[3] * P[5] * w 
end



@inline function function_PtoFy(P::AbstractArray{T}, Fy::AbstractArray{T},eos::Polytrope{T}) where T<:Real
    gam = sqrt(P[3]^2 + P[4]^2 + P[5]^2 + 1)
    w = eos.gamma * P[2] + P[1] 
    Fy[1] = P[1] * P[4]
    Fy[2] = - w *P[4] * gam
    Fy[3] = P[3] * P[4] * w 
    Fy[4] = P[4]^2 * w + (eos.gamma - 1) * P[2]
    Fy[5] = P[5] * P[4] * w 
end


@inline function function_PtoFz(P::AbstractArray{T}, Fy::AbstractArray{T},eos::Polytrope{T}) where T<:Real
    gam = sqrt(P[3]^2 + P[4]^2 + P[5]^2 + 1)
    w = eos.gamma * P[2] + P[1] 
    Fy[1] = P[1] * P[5]
    Fy[2] = - w *P[5] * gam
    Fy[3] = P[3] * P[5] * w 
    Fy[4] = P[4] * P[5]* w
    Fy[5] = w*P[5]^2 + (eos.gamma - 1) * P[2]
end


@inline function LU_dec!(flat_matrix::AbstractVector{T}, target::AbstractVector{T}, x::AbstractVector{T}) where T<:Real

    @inline function index(i, j)
        return (j - 1) * 5 + i
    end

    for k in 1:5
        for i in k+1:5
            flat_matrix[index(i, k)] /= flat_matrix[index(k, k)]
            for j in k+1:5
                flat_matrix[index(i, j)] -= flat_matrix[index(i, k)] * flat_matrix[index(k, j)]
            end
        end
    end

    # Forward substitution to solve L*y = target (reusing x for y)
    for i in 1:5
        x[i] = target[i]
        for j in 1:i-1
            x[i] -= flat_matrix[index(i, j)] * x[j]
        end
    end

    # Backward substitution to solve U*x = y
    for i in 5:-1:1
        for j in i+1:5
            x[i] -= flat_matrix[index(i, j)] * x[j]
        end
        x[i] /= flat_matrix[index(i, i)]
    end
end

@inline function LU_dec_pivot!(flat_matrix::AbstractVector{T}, target::AbstractVector{T}, x::AbstractVector{T}) where T<:Real
    @inline function index(i, j)
        # Column-major indexing for a 5x5 matrix
        return (j - 1) * 5 + i
    end

    # Use a statically sized mutable vector for the scaling factors to avoid dynamic allocation
    row_norm = MVector{5, T}(undef)

    # Compute the scaling vector for each row (row_norm[i] = 1 / max(|A[i, j]|))
    for i in 1:5
        absmax = zero(T)
        for j in 1:5
            temp = abs(flat_matrix[index(i, j)])
            if temp > absmax
                absmax = temp
            end
        end
        if absmax == 0
            error("LU_decompose(): row-wise singular matrix!")
        end
        row_norm[i] = one(T) / absmax
    end

    # LU decomposition using Crout's method with partial pivoting (scaled criterion)
    for k in 1:5
        # Find the pivot row by selecting the row with the largest scaled absolute value in column k
        pivot = k
        pivot_val = abs(flat_matrix[index(k, k)]) * row_norm[k]
        for i in k+1:5
            tmp = abs(flat_matrix[index(i, k)]) * row_norm[i]
            if tmp > pivot_val
                pivot_val = tmp
                pivot = i
            end
        end

        # If the pivot row is different from the current row, swap the rows in flat_matrix
        if pivot != k
            for j in 1:5
                tmp = flat_matrix[index(k, j)]
                flat_matrix[index(k, j)] = flat_matrix[index(pivot, j)]
                flat_matrix[index(pivot, j)] = tmp
            end
            # Swap the corresponding entries in the target vector
            tmp = target[k]
            target[k] = target[pivot]
            target[pivot] = tmp
            # Also swap the scaling factors for consistency
            tmp = row_norm[k]
            row_norm[k] = row_norm[pivot]
            row_norm[pivot] = tmp
        end

        # Perform elimination: update rows below the pivot row
        for i in k+1:5
            flat_matrix[index(i, k)] /= flat_matrix[index(k, k)]
            for j in k+1:5
                # Use fused multiply-add for improved accuracy
                flat_matrix[index(i, j)] = fma(-flat_matrix[index(i, k)], flat_matrix[index(k, j)], flat_matrix[index(i, j)])
            end
        end
    end

    # Forward substitution: solve L*y = target (store result in x)
    for i in 1:5
        x[i] = target[i]
        for j in 1:i-1
            x[i] = fma(-flat_matrix[index(i, j)], x[j], x[i])
        end
    end

    # Backward substitution: solve U*x = y
    for i in 5:-1:1
        for j in i+1:5
            x[i] = fma(-flat_matrix[index(i, j)], x[j], x[i])
        end
        x[i] /= flat_matrix[index(i, i)]
    end
end

@kernel inbounds = true function function_UtoP(@Const(U::AbstractArray{T}), P::AbstractArray{T},eos::Polytrope{T},n_iter::Int64,tol::T=1e-10) where T<:Real
    i, j, k = @index(Global, NTuple)
    il, jl, kl = @index(Local, NTuple)

    @uniform begin
        N,M,L = @groupsize()
        Nx,Ny,Nz = @ndrange()
    end
    
    Ploc = @localmem eltype(U) (5,N,M,L)
    Uloc = @localmem eltype(U) (5,N,M,L)

    
    for idx in 1:5
        Ploc[idx,il,jl,kl] = P[idx,i,j,k]
        Uloc[idx,il,jl,kl] = U[idx,i,j,k]
    end

    #buff_out = @MVector zeros(T,4)
    buff_out_t = @localmem eltype(U) (5,N,M,L)
    buff_out = @view buff_out_t[:,il,jl,kl]
    
    #buff_fun = @MVector zeros(T,4)
    buff_fun_t = @localmem eltype(U) (5,N,M,L)
    buff_fun = @view buff_fun_t[:,il,jl,kl]
    buff_jac = @MVector zeros(T,25)

    if i > 3 && i < Nx - 3 && j > 3 && j < Ny-3 && k > 3 && k < Nz-3
        for _ in 1:n_iter

            gam = sqrt(Ploc[3,il,jl,kl]^2 + Ploc[4,il,jl,kl]^2 +Ploc[5,il,jl,kl]^2 + 1)
            w = eos.gamma * Ploc[2,il,jl,kl] + Ploc[1,il,jl,kl] 
            
            buff_fun[1] = gam * Ploc[1,il,jl,kl] - Uloc[1,il,jl,kl]
            buff_fun[2] = (eos.gamma-1) * Ploc[2,il,jl,kl] - gam^2 * w - Uloc[2,il,jl,kl]
            buff_fun[3] = Ploc[3,il,jl,kl] * gam * w - Uloc[3,il,jl,kl]
            buff_fun[4] = Ploc[4,il,jl,kl] * gam * w - Uloc[4,il,jl,kl]
            buff_fun[5] = Ploc[5,il,jl,kl] * gam * w - Uloc[5,il,jl,kl]            



            buff_jac[1]  = gam
            buff_jac[6]  = 0   
            buff_jac[11] = Ploc[3,il,jl,kl] * Ploc[1,il,jl,kl]/gam
            buff_jac[16] = Ploc[4,il,jl,kl] * Ploc[1,il,jl,kl]/gam
            buff_jac[21] = Ploc[5,il,jl,kl] * Ploc[1,il,jl,kl]/gam  
    
            buff_jac[2]  = -gam^2
            buff_jac[7]  = eos.gamma*(-gam^2) + eos.gamma - 1 
            buff_jac[12] = -2*Ploc[3,il,jl,kl] * (w)
            buff_jac[17] = -2*Ploc[4,il,jl,kl] * (w)
            buff_jac[22] = -2*Ploc[5,il,jl,kl] * (w)
           
            buff_jac[3]  = Ploc[3,il,jl,kl] * gam  
            buff_jac[8]  = eos.gamma * Ploc[3,il,jl,kl] * gam 
            buff_jac[13] = Ploc[3,il,jl,kl] ^ 2 * w / gam + w * gam
            buff_jac[18] = Ploc[3,il,jl,kl] * Ploc[4,il,jl,kl] * w / gam
            buff_jac[23] = Ploc[3,il,jl,kl] * Ploc[5,il,jl,kl] * w / gam          
    
            buff_jac[4]  = Ploc[4,il,jl,kl] * gam
            buff_jac[9]  = eos.gamma * Ploc[4,il,jl,kl] * gam 
            buff_jac[14] = Ploc[3,il,jl,kl] * Ploc[4,il,jl,kl] * w / gam
            buff_jac[19] = Ploc[4,il,jl,kl]^2 * w / gam + w * gam
            buff_jac[24] = Ploc[4,il,jl,kl] * Ploc[5,il,jl,kl] * w / gam             

            buff_jac[5]  = Ploc[5,il,jl,kl] * gam
            buff_jac[10] = eos.gamma * Ploc[5,il,jl,kl] * gam 
            buff_jac[15] = Ploc[3,il,jl,kl] * Ploc[5,il,jl,kl] * w / gam
            buff_jac[20] = Ploc[4,il,jl,kl] * Ploc[5,il,jl,kl] * w / gam
            buff_jac[25] = Ploc[5,il,jl,kl] ^ 2 * w / gam + w * gam      
            
            
            LU_dec_pivot!(buff_jac,buff_fun,buff_out)

            if buff_out[1]^2 + buff_out[2]^2 + buff_out[3]^2 + buff_out[4]^2 +buff_out[5]^2 < tol ^ 2
                break
            end

            Ploc[1,il,jl,kl] = Ploc[1,il,jl,kl] - buff_out[1]
            Ploc[2,il,jl,kl] = Ploc[2,il,jl,kl] - buff_out[2]
            Ploc[3,il,jl,kl] = Ploc[3,il,jl,kl] - buff_out[3]
            Ploc[4,il,jl,kl] = Ploc[4,il,jl,kl] - buff_out[4]
            Ploc[5,il,jl,kl] = Ploc[5,il,jl,kl] - buff_out[5]
        end
    end
    @synchronize

    for idx in 1:5
        P[idx,i,j,k] = Ploc[idx,il,jl,kl]
    end
end
