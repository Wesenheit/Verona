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

@inline function LU_dec_5D!(flat_matrix::AbstractVector{T}, target::AbstractVector{T}, x::AbstractVector{T}) where T<:Real

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

@inline function LU_dec_2D!(flat_matrix::AbstractVector{T}, target::AbstractVector{T}, x::AbstractVector{T}) where T<:Real

    @inline function index(i, j)
        return (j - 1) * 2 + i
    end

    for k in 1:2
        for i in k+1:2
            flat_matrix[index(i, k)] /= flat_matrix[index(k, k)]
            for j in k+1:2
                flat_matrix[index(i, j)] -= flat_matrix[index(i, k)] * flat_matrix[index(k, j)]
            end
        end
    end

    for i in 1:2
        x[i] = target[i]
        for j in 1:i-1
            x[i] -= flat_matrix[index(i, j)] * x[j]
        end
    end

    for i in 2:-1:1
        for j in i+1:2
            x[i] -= flat_matrix[index(i, j)] * x[j]
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
    
 
    buff_out_t_2D = @localmem eltype(U) (2,N,M,L)
    buff_out_2D   = @view buff_out_t_2D[:,il,jl,kl]
    
    buff_fun_t_2D = @localmem eltype(U) (2,N,M,L)
    buff_fun_2D   = @view buff_fun_t_2D[:,il,jl,kl]
    buff_jac_2D   = @MVector zeros(T,4)

    buff_out_t_5D = @localmem eltype(U) (5,N,M,L)
    buff_out_5D   = @view buff_out_t_5D[:,il,jl,kl]
    
    buff_fun_t_5D = @localmem eltype(U) (5,N,M,L)
    buff_fun_5D   = @view buff_fun_t_5D[:,il,jl,kl]
    buff_jac_5D   = @MVector zeros(T,25)

    if i > 3 && i < Nx - 3 && j > 3 && j < Ny-3 && k > 3 && k < Nz-3
	
	#Conserved Variable
	D   = Uloc[1,il,jl,kl]
	Q_t = Uloc[2,il,jl,kl]
	Q_x = Uloc[3,il,jl,kl]
	Q_y = Uloc[4,il,jl,kl]
	Q_z = Uloc[5,il,jl,kl]

	#Useful Variables
	S²  	  = Q_x*Q_x + Q_y*Q_y + Q_z*Q_z
	γ 	      = sqrt(Ploc[3,il,jl,kl]^2 + Ploc[4,il,jl,kl]^2 +Ploc[5,il,jl,kl]^2 + 1)
	w_small   = Ploc[1,il,jl,kl] + (eos.gamma)*Ploc[2,il,jl,kl]
	v²        = ((Ploc[3,il,jl,kl])^2 + (Ploc[4,il,jl,kl])^2 + (Ploc[5,il,jl,kl])^2)/γ^2
	W 	      = w_small*γ^2
	W_max     = 1e30                #like in HARM        
	W_min     = sqrt(S²) * (1 - tol)     
    
    #Convergence
    convergence_1DW        = false
    convergence_2D         = false
    convergence_5D         = false
    convergence_bisection  = false

    #Initial condition, with v²<1
	while S² / W^2 >= 1 && W < W_max
	    W *= 10
	end

    #Additional useful values
    v²     = min(S² / W^2, 1 - tol)
    W_old  = W
    v²_old = v²

    #1DW METHOD
    #NEWTON-RAPHSON METHOD
    for _ in 1:n_iter
        
        if W < W_min
            W = W_min
        end

        v²  = S² / W^2
        
        if v² < 0.0
            v² = 0.0
        elseif v² > 1 - tol
            v² = 1 - tol
        end	        
        
        buff_fun = Q_t + W - ((eos.gamma - 1) / eos.gamma) * (W * (1 - v²) - D * sqrt(1 - v²))
        buff_jac = 1 - ((eos.gamma - 1) / eos.gamma) * (1 - v²) - ((eos.gamma - 1) / eos.gamma) * (W * (2 * S² / W^3) - D * (S² / (W^3 * sqrt(1 - v²))))
        
        ΔW = buff_fun / buff_jac
        W_proposed = W - ΔW

        if W_proposed < W_min
            W = 0.5 * (W + W_min)
        else
            W = W_proposed
        end
   
    
        if ΔW < 0
            ΔW = -ΔW
        end  

        if ΔW^2 < tol^2
            convergence_1DW = true
            break
        end
    
    end

    if !convergence_1DW
        #Useful Variables for 2D
        S²  	  = Q_x*Q_x + Q_y*Q_y + Q_z*Q_z
        γ 	      = sqrt(Ploc[3,il,jl,kl]^2 + Ploc[4,il,jl,kl]^2 +Ploc[5,il,jl,kl]^2 + 1)
        w_small   = Ploc[1,il,jl,kl] + (eos.gamma)*Ploc[2,il,jl,kl]
        v²        = ((Ploc[3,il,jl,kl])^2 + (Ploc[4,il,jl,kl])^2 + (Ploc[5,il,jl,kl])^2)/γ^2
        W 	      = w_small*γ^2
        W_max     = 1e30 #like in HARM        
        W_min     = sqrt(S²) * (1 - tol)     
        
        #Initial condition, with v²<1
        while S² / W^2 >= 1 && W < W_max
            W *= 10
        end
        #Additional useful values
        v²     = min(S² / W^2, 1 - tol)
        W_old  = W
        v²_old = v²


        #2D METHOD
        #NEWTON-RAPHSON METHOD
        for _ in 1:n_iter
            W_old  = W
            v²_old = v²


            buff_fun_2D[1] = v² - S² / W^2
            buff_fun_2D[2] = Q_t + W - ((eos.gamma - 1) / eos.gamma) * (W * (1 - v²) - D * sqrt(1 - v²))

            buff_jac_2D[1] = 2 * S² / W^3
            buff_jac_2D[2] = 1 - ((eos.gamma - 1) / eos.gamma) * (1 - v²)
            buff_jac_2D[3] = 1
            buff_jac_2D[4] = ((eos.gamma - 1) / eos.gamma) * (W - D / (2 * sqrt(1 - v²)))
            
            LU_dec_2D!(buff_jac_2D, buff_fun_2D, buff_out_2D)
            
            ΔW  = buff_out_2D[1]
            Δv² = buff_out_2D[2]
            α   = 1.0
    
            converged = false
            
            for _ in 1:10
                W_candidate  = (sqrt(W - α * ΔW)^2)
                v²_candidate = v² - α * Δv²

                if W_candidate <= 0 || W_candidate > W_max
                    W_candidate = W_old
                end

                if v²_candidate < 0
                    v²_candidate = 0.0
                elseif v²_candidate >= 1
                    v²_candidate = 1.0 - tol
                end

                r1 = v²_candidate - S² / W_candidate^2
                r2 = Q_t + W_candidate - ((eos.gamma - 1) / eos.gamma) * (W_candidate * (1 - v²_candidate) - D * sqrt(1 - v²_candidate))
                err_candidate = r1^2 + r2^2

                r1_old = v² - S² / W^2
                r2_old = Q_t + W - ((eos.gamma - 1) / eos.gamma) * (W * (1 - v²) - D * sqrt(1 - v²))
                err_old = r1_old^2 + r2_old^2

                if err_candidate <= err_old
                    W = W_candidate
                    v² = v²_candidate
                    converged = true
                    break
                else
                    α *= 0.5
                end
            end
            
            if !converged
                W  = W_old
                v² = v²_old
            end

            relW  = sqrt((W - W_old)^2) / max(abs(W_old), tol^2)
            relv² = sqrt((v² - v²_old)^2) / max(abs(v²_old), tol^2)
            if (relW + relv²) < tol
                convergence_2D = true
                break
            end
        end
    end
    
    if convergence_1DW || convergence_2D 
        v² = S² / W^2
        γ  = 1/sqrt(1-v²)
        Ploc[1,il,jl,kl] = D / γ
        Ploc[2,il,jl,kl] = (W / γ^2 - D / γ) / eos.gamma
        Ploc[3,il,jl,kl] = γ*Q_x/W
        Ploc[4,il,jl,kl] = γ*Q_y/W
        Ploc[5,il,jl,kl] = γ*Q_z/W
    end

    #5D METHOD
    #NEWTON-RAPHSON METHOD
    if !convergence_1DW && !convergence_2D
        for _ in 1:n_iter

            gam = sqrt(Ploc[3,il,jl,kl]^2 + Ploc[4,il,jl,kl]^2 +Ploc[5,il,jl,kl]^2 + 1)
            w = eos.gamma * Ploc[2,il,jl,kl] + Ploc[1,il,jl,kl] 
            
            buff_fun_5D[1] = gam * Ploc[1,il,jl,kl] - Uloc[1,il,jl,kl]
            buff_fun_5D[2] = (eos.gamma-1) * Ploc[2,il,jl,kl] - gam^2 * w - Uloc[2,il,jl,kl]
            buff_fun_5D[3] = Ploc[3,il,jl,kl] * gam * w - Uloc[3,il,jl,kl]
            buff_fun_5D[4] = Ploc[4,il,jl,kl] * gam * w - Uloc[4,il,jl,kl]
            buff_fun_5D[5] = Ploc[5,il,jl,kl] * gam * w - Uloc[5,il,jl,kl]            

            buff_jac_5D[1]  = gam
            buff_jac_5D[6]  = 0   
            buff_jac_5D[11] = Ploc[3,il,jl,kl] * Ploc[1,il,jl,kl]/gam
            buff_jac_5D[16] = Ploc[4,il,jl,kl] * Ploc[1,il,jl,kl]/gam
            buff_jac_5D[21] = Ploc[5,il,jl,kl] * Ploc[1,il,jl,kl]/gam  
    
            buff_jac_5D[2]  = -gam^2
            buff_jac_5D[7]  = eos.gamma*(-gam^2) + eos.gamma - 1 
            buff_jac_5D[12] = -2*Ploc[3,il,jl,kl] * (w)
            buff_jac_5D[17] = -2*Ploc[4,il,jl,kl] * (w)
            buff_jac_5D[22] = -2*Ploc[5,il,jl,kl] * (w)
           
            buff_jac_5D[3]  = Ploc[3,il,jl,kl] * gam  
            buff_jac_5D[8]  = eos.gamma * Ploc[3,il,jl,kl] * gam 
            buff_jac_5D[13] = Ploc[3,il,jl,kl] ^ 2 * w / gam + w * gam
            buff_jac_5D[18] = Ploc[3,il,jl,kl] * Ploc[4,il,jl,kl] * w / gam
            buff_jac_5D[23] = Ploc[3,il,jl,kl] * Ploc[5,il,jl,kl] * w / gam          
    
            buff_jac_5D[4]  = Ploc[4,il,jl,kl] * gam
            buff_jac_5D[9]  = eos.gamma * Ploc[4,il,jl,kl] * gam 
            buff_jac_5D[14] = Ploc[3,il,jl,kl] * Ploc[4,il,jl,kl] * w / gam
            buff_jac_5D[19] = Ploc[4,il,jl,kl]^2 * w / gam + w * gam
            buff_jac_5D[24] = Ploc[4,il,jl,kl] * Ploc[5,il,jl,kl] * w / gam             

            buff_jac_5D[5]  = Ploc[5,il,jl,kl] * gam
            buff_jac_5D[10] = eos.gamma * Ploc[5,il,jl,kl] * gam 
            buff_jac_5D[15] = Ploc[3,il,jl,kl] * Ploc[5,il,jl,kl] * w / gam
            buff_jac_5D[20] = Ploc[4,il,jl,kl] * Ploc[5,il,jl,kl] * w / gam
            buff_jac_5D[25] = Ploc[5,il,jl,kl] ^ 2 * w / gam + w * gam      
            
            
            LU_dec_5D!(buff_jac_5D,buff_fun_5D,buff_out_5D)

            if buff_out_5D[1]^2 + buff_out_5D[2]^2 + buff_out_5D[3]^2 + buff_out_5D[4]^2 +buff_out_5D[5]^2 < tol ^ 2
                convergence_5D = true
                break
            end

            Ploc[1,il,jl,kl] = Ploc[1,il,jl,kl] - buff_out_5D[1]
            Ploc[2,il,jl,kl] = Ploc[2,il,jl,kl] - buff_out_5D[2]
            Ploc[3,il,jl,kl] = Ploc[3,il,jl,kl] - buff_out_5D[3]
            Ploc[4,il,jl,kl] = Ploc[4,il,jl,kl] - buff_out_5D[4]
            Ploc[5,il,jl,kl] = Ploc[5,il,jl,kl] - buff_out_5D[5]
        end
    end
    

    if !convergence_1DW && !convergence_2D && !convergence_5D
        fun_min = Q_t + W_min - ((eos.gamma - 1) / eos.gamma) * (W_min * (1 - S² / W_min^2) - D * sqrt(1 - S² / W_min^2))
        fun_max = Q_t + W_max - ((eos.gamma - 1) / eos.gamma) * (W_max * (1 - S² / W_max^2) - D * sqrt(1 - S² / W_max^2))
        
        if fun_min*fun_max < 0 #It is assumed that the root is beetween W_min and W_max - should be!
            while convergence_bisection == false
                fun_min = Q_t + W_min - ((eos.gamma - 1) / eos.gamma) * (W_min * (1 - S² / W_min^2) - D * sqrt(1 - S² / W_min^2))
                fun_max = Q_t + W_max - ((eos.gamma - 1) / eos.gamma) * (W_max * (1 - S² / W_max^2) - D * sqrt(1 - S² / W_max^2))
                
                W_mid = 0.5 * (W_min + W_max)
                fun_mid = Q_t + W_mid - ((eos.gamma - 1) / eos.gamma) * (W_mid * (1 - S² / W_mid^2) - D * sqrt(1 - S² / W_mid^2))
                
                if fun_mid^2 < (tol)^2
                    convergence_bisection = true
                    W = W_mid
                    break
                end

                if fun_min * fun_mid < 0
                    W_max = W_mid
                else
                    W_min = W_mid
                end   
            end
        end            
    if convergence_bisection 
        v² = S² / W^2
        γ  = 1/sqrt(1-v²)
        Ploc[1,il,jl,kl] = D / γ
        Ploc[2,il,jl,kl] = (W / γ^2 - D / γ) / eos.gamma
        Ploc[3,il,jl,kl] = γ*Q_x/W
        Ploc[4,il,jl,kl] = γ*Q_y/W
        Ploc[5,il,jl,kl] = γ*Q_z/W
    end
    end
    
    if !convergence_1DW && !convergence_2D && !convergence_5D && !convergence_bisection
    	for idx in 1:5
        	Ploc[idx,i,j,k] = P[idx,il,jl,kl]
    	end    
    end
    
    end
    @synchronize
    for idx in 1:5
    	P[idx,i,j,k] = Ploc[idx,il,jl,kl]
    end
end
