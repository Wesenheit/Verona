using LinearAlgebra
using StaticArrays
using Base.Threads
using Base.Iterators
using KernelAbstractions
using CUDA
using MPI
using HDF5
# Used scheme
# U - conserved varaibles
# U1 = rho ut - mass conservation
# U2 = T^t_t - energy conservation
# U3 = T^t_x - momentum conservation x
# U4 = T^t_y - momentum conservation y
 
# P - primitive variables
# P1 = rho - density
# P2 = u - energy density
# P3 = ux four-velocity in x
# P4 = uy four-velocity in y

function local_to_global(i,p,Size,MPI)
    if p == 0
        return i
    elseif p > 0 && p < MPI-1 && (px < 4 || px > Size-3) 
        return 0
    else
        return i + p * (Size - 6)
    end
end


abstract type VeronaArr{T} end


mutable struct ParVector2D{T <:Real} <: VeronaArr{T}
    # Parameter Vector
    arr::Array{T,3}
    size_X::Int64
    size_Y::Int64
    function ParVector2D{T}(Nx,Ny) where {T}
        arr = zeros(T,4,Nx + 6,Ny + 6)
        new(arr,Nx + 6 ,Ny + 6)
    end
    function ParVector2D{T}(arr::VeronaArr{T}) where {T}
        new(Array{T}(arr.arr),arr.size_X,arr.size_Y)
    end
end

mutable struct CuParVector2D{T <:Real} <: VeronaArr{T}
    # Parameter Vector
    arr::CuArray{T}
    size_X::Int64
    size_Y::Int64
    function CuParVector2D{T}(arr::VeronaArr{T}) where {T}
        new(CuArray{T}(arr.arr),arr.size_X,arr.size_Y)
    end

    function CuParVector2D{T}(Nx::Int64,Ny::Int64) where {T}
        new(CuArray{T}(zeros(T,4,Nx + 6,Ny + 6)), Nx + 6,Ny + 6)
    end
end

function VectorLike(X::VeronaArr{T}) where T
    if typeof(X.arr) <: CuArray
        return CuParVector2D{T}(X.size_X-6,X.size_Y-6)
    else
        return ParVector2D{T}(X.size_X-6,X.size_Y-6)
    end
end

@kernel inbounds = true function kernel_PtoU(@Const(P::AbstractArray{T}), U::AbstractArray{T},eos::Polytrope{T}) where T
    i, j = @index(Global, NTuple)    
    begin
        gam = sqrt(P[3,i,j]^2 + P[4,i,j]^2 + 1)
        w = eos.gamma * P[2,i,j] + P[1,i,j] 
        U[1,i,j] = gam * P[1,i,j]
        U[2,i,j] = (eos.gamma-1) * P[2,i,j] - gam^2 * w
        U[3,i,j] = P[3,i,j] * gam * w
        U[4,i,j] = P[4,i,j] * gam * w
    end
end


@inline function function_PtoU(P::AbstractVector{T}, U::AbstractVector{T},eos::Polytrope{T}) where T
    gam = sqrt(P[3]^2 + P[4]^2 + 1)
    w = eos.gamma * P[2] + P[1] 
    U[1] = gam * P[1]
    U[2] = (eos.gamma-1) * P[2] - gam^2 * w
    U[3] = P[3] * gam * w
    U[4] = P[4] * gam * w
end


@inline function function_PtoFx(P::AbstractVector{T}, Fx::AbstractVector{T},eos::Polytrope{T}) where T
    gam = sqrt(P[3]^2 + P[4]^2 + 1)
    w = eos.gamma * P[2] + P[1] 

    Fx[1] = P[1] * P[3]
    Fx[2] = - w *P[3] * gam
    Fx[3] = P[3]^2 * w + (eos.gamma - 1) * P[2]
    Fx[4] = P[3] * P[4] * w 
end


@inline function function_PtoFy(P::AbstractArray{T}, Fy::AbstractArray{T},eos::Polytrope{T}) where T
    gam = sqrt(P[3]^2 + P[4]^2 + 1)
    w = eos.gamma * P[2] + P[1] 
    Fy[1] = P[1] * P[4]
    Fy[2] = - w *P[4] * gam
    Fy[3] = P[3] * P[4] * w 
    Fy[4] = P[4]^2 * w + (eos.gamma - 1) * P[2]
end

@inline function LU_dec_pivot!(flat_matrix::AbstractVector{T}, target::AbstractVector{T}, x::AbstractVector{T}) where T
    @inline function index(i, j)
        # Column-major indexing for a 4x4 matrix
        return (j - 1) * 4 + i
    end

    # Use a statically sized mutable vector for scaling factors to avoid dynamic allocation
    row_norm = MVector{4, T}(undef)

    # Compute the scaling factor for each row (row_norm[i] = 1 / max(|A[i, j]|))
    for i in 1:4
        absmax = zero(T)
        for j in 1:4
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

    # Perform LU decomposition using Crout's method with partial pivoting (scaled criterion)
    for k in 1:4
        # Find the pivot row by selecting the row with the largest scaled absolute value in column k
        pivot = k
        pivot_val = abs(flat_matrix[index(k, k)]) * row_norm[k]
        for i in k+1:4
            tmp = abs(flat_matrix[index(i, k)]) * row_norm[i]
            if tmp > pivot_val
                pivot_val = tmp
                pivot = i
            end
        end

        # If the pivot row is different from the current row, swap the rows in flat_matrix
        if pivot != k
            for j in 1:4
                tmp = flat_matrix[index(k, j)]
                flat_matrix[index(k, j)] = flat_matrix[index(pivot, j)]
                flat_matrix[index(pivot, j)] = tmp
            end
            # Also swap the corresponding entries in the target vector
            tmp = target[k]
            target[k] = target[pivot]
            target[pivot] = tmp
            # Also swap the scaling factors for consistency
            tmp = row_norm[k]
            row_norm[k] = row_norm[pivot]
            row_norm[pivot] = tmp
        end

        # Perform elimination: update rows below the pivot row
        for i in k+1:4
            flat_matrix[index(i, k)] /= flat_matrix[index(k, k)]
            for j in k+1:4
                # Use fused multiply-add for improved accuracy
                flat_matrix[index(i, j)] = fma(-flat_matrix[index(i, k)], flat_matrix[index(k, j)], flat_matrix[index(i, j)])
            end
        end
    end

    # Forward substitution: solve L*y = target (store result in x)
    for i in 1:4
        x[i] = target[i]
        for j in 1:i-1
            x[i] = fma(-flat_matrix[index(i, j)], x[j], x[i])
        end
    end

    # Backward substitution: solve U*x = y
    for i in 4:-1:1
        for j in i+1:4
            x[i] = fma(-flat_matrix[index(i, j)], x[j], x[i])
        end
        x[i] /= flat_matrix[index(i, i)]
    end
end



@inline function LU_dec!(flat_matrix::AbstractVector{T}, target::AbstractVector{T}, x::AbstractVector{T}) where T

    @inline function index(i, j)
        return (j - 1) * 4 + i
    end

    for k in 1:4
        for i in k+1:4
            flat_matrix[index(i, k)] /= flat_matrix[index(k, k)]
            for j in k+1:4
                flat_matrix[index(i, j)] -= flat_matrix[index(i, k)] * flat_matrix[index(k, j)]
            end
        end
    end

    # Forward substitution to solve L*y = target (reusing x for y)
    for i in 1:4
        x[i] = target[i]
        for j in 1:i-1
            x[i] -= flat_matrix[index(i, j)] * x[j]
        end
    end

    # Backward substitution to solve U*x = y
    for i in 4:-1:1
        for j in i+1:4
            x[i] -= flat_matrix[index(i, j)] * x[j]
        end
        x[i] /= flat_matrix[index(i, i)]
    end
end

@kernel inbounds = true function function_UtoP(@Const(U::AbstractArray{T}), P::AbstractArray{T},eos::Polytrope{T},n_iter::Int64,tol::T=1e-10) where T
    i, j = @index(Global, NTuple)
    il, jl = @index(Local, NTuple)
    i = Int32(i)
    j = Int32(j)
    il = Int32(il)
    jl = Int32(jl)
    @uniform begin
        Nx,Ny = @ndrange()
        N,M = @groupsize()
    end
    
    Ploc = @localmem eltype(U) (4,N,M)
    Uloc = @localmem eltype(U) (4,N,M)

    
    for idx in 1:4
        Ploc[idx,il,jl] = P[idx,i,j]
        Uloc[idx,il,jl] = U[idx,i,j]
    end

    #buff_out = @MVector zeros(T,4)
    buff_out_t = @localmem eltype(U) (4,N,M)
    buff_out = @view buff_out_t[:,il,jl]
    
    #buff_fun = @MVector zeros(T,4)
    buff_fun_t = @localmem eltype(U) (4,N,M)
    buff_fun = @view buff_fun_t[:,il,jl]
    buff_jac = @MVector zeros(T,16)

    if i > 3 && i < Nx - 2 && j > 3 && j < Ny-2
        for _ in 1:n_iter
            gam = sqrt(Ploc[3,il,jl]^2 + Ploc[4,il,jl]^2 + 1)
            w = eos.gamma * Ploc[2,il,jl] + Ploc[1,il,jl] 
            
            buff_fun[1] = gam * Ploc[1,il,jl] - Uloc[1,il,jl]
            buff_fun[2] = (eos.gamma-1) * Ploc[2,il,jl] - gam^2 * w - Uloc[2,il,jl]
            buff_fun[3] = Ploc[3,il,jl] * gam * w - Uloc[3,il,jl]
            buff_fun[4] = Ploc[4,il,jl] * gam * w - Uloc[4,il,jl]

            buff_jac[1] = gam
            buff_jac[5] = 0
            buff_jac[9] = Ploc[1,il,jl] * Ploc[3,il,jl] / gam
            buff_jac[13] = Ploc[1,il,jl] * Ploc[4,il,jl] / gam
    
            buff_jac[2] = - gam^2
            buff_jac[6] = (eos.gamma - 1) - gam ^2 * eos.gamma
            buff_jac[10] = -2 * Ploc[3,il,jl] * w
            buff_jac[14] = -2 * Ploc[4,il,jl] * w
    
            buff_jac[3] = gam * Ploc[3,il,jl]
            buff_jac[7] = gam * Ploc[3,il,jl] * eos.gamma
            buff_jac[11] = (2 * Ploc[3,il,jl]^2 + Ploc[4,il,jl]^2 + 1) / gam * w
            buff_jac[15] = Ploc[3,il,jl] * Ploc[4,il,jl] / gam * w
    
            buff_jac[4] = gam * Ploc[4,il,jl]
            buff_jac[8] = gam * Ploc[4,il,jl] * eos.gamma
            buff_jac[12] = Ploc[3,il,jl] * Ploc[4,il,jl] / gam * w
            buff_jac[16] = (2 * Ploc[4,il,jl]^2 + Ploc[3,il,jl] ^ 2 + 1) / gam * w                        
            
            LU_dec_pivot!(buff_jac,buff_fun,buff_out)

            if sqrt(buff_out[1]^2 + buff_out[2]^2 + buff_out[3]^2 + buff_out[4]^2) < tol
                break
            end
            Ploc[1,il,jl] = Ploc[1,il,jl] - buff_out[1]
            Ploc[2,il,jl] = Ploc[2,il,jl] - buff_out[2]
            Ploc[3,il,jl] = Ploc[3,il,jl] - buff_out[3]
            Ploc[4,il,jl] = Ploc[4,il,jl] - buff_out[4]
        end
    end

    for idx in 1:4
        P[idx,i,j] = Ploc[idx,il,jl]
    end
end
