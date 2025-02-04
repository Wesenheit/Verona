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
# U5 = T^t_z - momentum conservation z
 
# P - primitive variables
# P1 = rho - density
# P2 = u - energy density
# P3 = ux four-velocity in x
# P4 = uy four-velocity in y
# P5 = uz four-velocity in z

function local_to_global(i,px,Size_X,MPI_X)
    if px == 0
        return i
    elseif px > 0 && px < MPI_X-1 && (px < 4 || px > Size_X-3) 
        return 0
    else
        return i + px * (Size_X - 6)
    end
end


abstract type FlowArr{T} end


mutable struct ParVector2D{T <:Real} <: FlowArr{T}
    # Parameter Vector
    arr::Array{T,3}
    size_X::Int64
    size_Y::Int64
    function ParVector2D{T}(Nx,Ny) where {T}
        arr = zeros(T,4,Nx + 6,Ny + 6)
        new(arr,Nx + 6 ,Ny + 6)
    end
    function ParVector2D{T}(arr::FlowArr{T}) where {T}
        new(Array{T}(arr.arr),arr.size_X,arr.size_Y)
    end
end

mutable struct CuParVector2D{T <:Real} <: FlowArr{T}
    # Parameter Vector
    arr::CuArray{T}
    size_X::Int32
    size_Y::Int32
    function CuParVector2D{T}(arr::FlowArr{T}) where {T}
        new(CuArray{T}(arr.arr),arr.size_X,arr.size_Y)
    end

    function CuParVector2D{T}(Nx::Int64,Ny::Int64) where {T}
        new(CuArray{T}(zeros(T,4,Nx + 6,Ny + 6)), Nx + 6,Ny + 6)
    end
end

function VectorLike(X::FlowArr{T}) where T
    if typeof(X.arr) <: CuArray
        return CuParVector2D{T}(X.size_X-6,X.size_Y-6)
    else
        return ParVector2D{T}(X.size_X-6,X.size_Y-6)
    end
end

@kernel function kernel_PtoU(@Const(P::AbstractArray{T}), U::AbstractArray{T},eos::Polytrope{T}) where T
    i, j, k = @index(Global, NTuple)    
    @inbounds begin
        gam = sqrt(P[3,i,j,k]^2 + P[4,i,j,k]^2 + P[5,i,j,k]^2 + 1)  #gam = u⁰
        w = eos.gamma * P[2,i,j,k] + P[1,i,j,k]                     # ρ + u + p = ρ + eos.gamma*u
        U[1,i,j,k] = gam * P[1,i,j,k]
        U[2,i,j,k] = (eos.gamma-1) * P[2,i,j,k] - gam^2 * w 
        U[3,i,j,k] = P[3,i,j,k] * gam * w
        U[4,i,j,k] = P[4,i,j,k] * gam * w
        U[5,i,j,k] = P[5,i,j,k] * gam * w        
    end
end


@inline function function_PtoU(P::AbstractVector{T}, U::AbstractVector{T},eos::Polytrope{T}) where T
    gam = sqrt(P[3]^2 + P[4]^2 +P[5]^2 + 1) #gam = u⁰
    w = eos.gamma * P[2] + P[1] 
    U[1] = gam * P[1]
    U[2] = (eos.gamma-1) * P[2] - gam^2 * w
    U[3] = P[3] * gam * w
    U[4] = P[4] * gam * w
    U[5] = P[5] * gam * w
end


@inline function function_PtoFx(P::AbstractVector{T}, Fx::AbstractVector{T},eos::Polytrope{T}) where T
    gam = sqrt(P[3]^2 + P[4]^2 + P[5]^2 + 1)
    w = eos.gamma * P[2] + P[1] 
    Fx[1] = P[1] * P[3]
    Fx[2] = - w *P[3] * gam
    Fx[3] = P[3]^2 * w + (eos.gamma - 1) * P[2]
    Fx[4] = P[3] * P[4] * w 
    Fx[5] = P[3] * P[5] * w 
end



@inline function function_PtoFy(P::AbstractArray{T}, Fy::AbstractArray{T},eos::Polytrope{T}) where T
    gam = sqrt(P[3]^2 + P[4]^2 + P[5]^2 + 1)
    w = eos.gamma * P[2] + P[1] 
    Fy[1] = P[1] * P[4]
    Fy[2] = - w *P[4] * gam
    Fy[3] = P[3] * P[4] * w 
    Fy[4] = P[4]^2 * w + (eos.gamma - 1) * P[2]
    Fy[5] = P[5] * P[4] * w 
end


@inline function function_PtoFz(P::AbstractArray{T}, Fy::AbstractArray{T},eos::Polytrope{T}) where T
    gam = sqrt(P[3]^2 + P[4]^2 + P[5]^2 + 1)
    w = eos.gamma * P[2] + P[1] 
    Fy[1] = P[1] * P[5]
    Fy[2] = - w *P[5] * gam
    Fy[3] = P[3] * P[5] * w 
    Fy[4] = P[4] * P[5]* w
    Fy[5] = w*P[5]^2 + (eos.gamma - 1) * P[2]
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
    N = @uniform @groupsize()[1]
    M = @uniform @groupsize()[2]
    
    Ploc = @localmem eltype(U) (4,N,M)
    Uloc = @localmem eltype(U) (4,N,M)

    
    for idx in 1:4
        Ploc[idx,il,jl] = P[idx,i,j]
        Uloc[idx,il,jl] = U[idx,i,j]
    end
    @synchronize

    #buff_out = @MVector zeros(T,4)
    buff_out_t = @localmem eltype(U) (4,N,M)
    buff_out = @view buff_out_t[:,il,jl]
    
    #buff_fun = @MVector zeros(T,4)
    buff_fun_t = @localmem eltype(U) (4,N,M)
    buff_fun = @view buff_fun_t[:,il,jl]
    buff_jac = @MVector zeros(T,16)
    Nx,Ny = @ndrange()

    if i > 3 && i < Nx - 3 && j > 3 && j < Ny-3
        for _ in 1:n_iter
            gam = sqrt(Ploc[3,il,jl]^2 + Ploc[4,il,jl]^2 +Ploc[5,il,jl]^2 + 1)
            w = eos.gamma * Ploc[2,il,jl] + Ploc[1,il,jl] 
            
            buff_fun[1] = gam * Ploc[1,il,jl] - Uloc[1,il,jl]
            buff_fun[2] = (eos.gamma-1) * Ploc[2,il,jl] - gam^2 * w - Uloc[2,il,jl]
            buff_fun[3] = Ploc[3,il,jl] * gam * w - Uloc[3,il,jl]
            buff_fun[4] = Ploc[4,il,jl] * gam * w - Uloc[4,il,jl]
            buff_fun[5] = Ploc[5,il,jl] * gam * w - Uloc[5,il,jl]            



            buff_jac[1]  = gam
            buff_jac[6]  = 0   
            buff_jac[11] = Ploc[3,il,jl]*Ploc[1,il,jl]/gam
            buff_jac[16] = Ploc[4,il,jl]*Ploc[1,il,jl]/gam
            buff_jac[21] = Ploc[5,il,jl]*Ploc[1,il,jl]/gam  
    
            buff_jac[2]  = -gam^2
            buff_jac[7]  = eos.gamma*(-gam^2) + eos.gamma - 1 
            buff_jac[12] = -2*Ploc[3,il,jl]*(w)
            buff_jac[17] = -2*Ploc[4,il,jl]*(w)
            buff_jac[22] = -2*Ploc[5,il,jl]*(w)
           
            buff_jac[3]  = Ploc[3,il,jl]*gam  
            buff_jac[8]  = eos.gamma*Ploc[3,il,jl]*gam 
            buff_jac[13] = Ploc[3,il,jl]^2*(w)/gam + (w)*gam
            buff_jac[18] = Ploc[3,il,jl]*Ploc[4,il,jl]*(w)/gam
            buff_jac[23] = Ploc[3,il,jl]*Ploc[5,il,jl]*(w)/gam          
    
            buff_jac[4]  = Ploc[4,il,jl]*gam
            buff_jac[9]  = eos.gamma*Ploc[4,il,jl]*gam 
            buff_jac[14] = Ploc[3,il,jl]*Ploc[4,il,jl]*(w)/gam
            buff_jac[19] = Ploc[4,il,jl]^2*(w)/gam + (w)*gam
            buff_jac[24] = Ploc[4,il,jl]*Ploc[5,il,jl]*(w)/gam             

            buff_jac[5]  = Ploc[5,il,jl]*gam
            buff_jac[10] = eos.gamma*Ploc[5,il,jl]*gam 
            buff_jac[15] = Ploc[3,il,jl]*Ploc[5,il,jl]*(w)/gam
            buff_jac[20] = Ploc[4,il,jl]*Ploc[5,il,jl]*(w)/gam
            buff_jac[25] = Ploc[5,il,jl]^2*(w)/gam + (w)*gam      
            
            
            LU_dec!(buff_jac,buff_fun,buff_out)

            if sqrt(buff_out[1]^2 + buff_out[2]^2 + buff_out[3]^2 + buff_out[4]^2 +buff_out[5]^2) < tol
                break
            end
            Ploc[1,il,jl] = Ploc[1,il,jl] - buff_out[1]
            Ploc[2,il,jl] = Ploc[2,il,jl] - buff_out[2]
            Ploc[3,il,jl] = Ploc[3,il,jl] - buff_out[3]
            Ploc[4,il,jl] = Ploc[4,il,jl] - buff_out[4]
            Ploc[5,il,jl] = Ploc[5,il,jl] - buff_out[5]
        end
    end
    @synchronize
    for idx in 1:5
        P[idx,i,j] = Ploc[idx,il,jl]
    end
end
