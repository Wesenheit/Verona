using LinearAlgebra
using StaticArrays
using Base.Threads
using Base.Iterators
using KernelAbstractions
using CUDA
using MPI
using HDF5

abstract type VeronaArr{T} end

mutable struct ParVector3D{T <:Real} <: VeronaArr{T}
    # Parameter Vector
    arr::Array{T,4}
    size_X::Int64
    size_Y::Int64
    size_Z::Int64
    function ParVector3D{T}(Nx,Ny,Nz) where {T}
        arr = zeros(T, 5, Nx + 6, Ny + 6, Nz + 6)
        new(arr, Nx + 6 , Ny + 6, Nz + 6)
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

@kernel inbounds = true function kernel_PtoU(@Const(P::AbstractArray{T}), P_stagger::AbstractArray{T}, U::AbstractArray{T}, eos::Polytrope{T}) where T<:Real
    i, j, k = @index(Global, NTuple)    
    begin
        # Primitive variables
        ρ  = P[1,i,j,k]                                       # Rest-mass density
	u  = P[2,i,j,k]                                       # Specific internal energy 
        u¹ = P[3,i,j,k]                                       # Contravariant four-velocity in the x-direction
        u² = P[4,i,j,k]                                       # Contravariant four-velocity in the y-direction
        u³ = P[5,i,j,k]                                       # Contravariant four-velocity in the z-direction     
        B¹ = 1/2 *(P_stagger[1,i+1,j,k] + P_stagger[1,i,j,k]) # Magnetic field component in the x-direction
        B² = 1/2 *(P_stagger[2,i,j+1,k] + P_stagger[2,i,j,k]) # Magnetic field component in the y-direction
        B³ = 1/2 *(P_stagger[3,i,j,k+1] + P_stagger[3,i,j,k]) # Magnetic field component in the z-direction
        
        # Contravariant time component of the four-velocity
        u⁰ = sqrt(u¹*u¹ + u²*u² + u³*u³ + 1)

        # Covariant components of the four-velocity
        u₀ = -u⁰ 
        u₁ =  u¹
        u₂ =  u²
        u₃ =  u³
        
        # Contravariant components of the magnetic four-vector
        b⁰ = B¹*u₁ + B²*u₂ + B³*u₃
        b¹ = (B¹ + b⁰*u¹)/u⁰
        b² = (B² + b⁰*u²)/u⁰
        b³ = (B³ + b⁰*u³)/u⁰	

        # Covariant components of the magnetic four-vector
        b₀ = -b⁰
        b₁ =  b¹
        b₂ =  b²
        b₃ =  b³	

        # Magnetic four-vector contraction            
        bsq = b⁰*b₀ + b¹*b₁ + b²*b₂ + b³*b₃

        #Total enthalpy
        enthalpy = ρ + eos.gamma*u + bsq
        
        # Conserved quantities in relativistic magnetohydrodynamics 
        U[1,i,j,k] = ρ*u⁰                                                  # Conserved mass density (D)
        U[2,i,j,k] = enthalpy*u⁰*u₀ + (eos.gamma - 1)*u + bsq/2 - b⁰*b₀    # Energy density (Q₀)
        U[3,i,j,k] = enthalpy*u⁰*u₁ - b⁰*b₁                                # Conserved momentum density in the x-direction (Q₁)    
        U[4,i,j,k] = enthalpy*u⁰*u₂ - b⁰*b₂                                # Conserved momentum density in the y-direction (Q₂)
        U[5,i,j,k] = enthalpy*u⁰*u₃ - b⁰*b₃                                # Conserved momentum density in the z-direction (Q₃)
    end
end


@inline function function_PtoU(P::AbstractVector{T}, U::AbstractVector{T}, eos::Polytrope{T}) where T<:Real
    
    # Primitive variables
    ρ  = P[1] # Rest-mass density
    u  = P[2] # Specific internal energy 
    u¹ = P[3] # Contravariant four-velocity in the x-direction
    u² = P[4] # Contravariant four-velocity in the y-direction
    u³ = P[5] # Contravariant four-velocity in the z-direction     
    B¹ = P[6] # Magnetic field component in the x-direction
    B² = P[7] # Magnetic field component in the y-direction
    B³ = P[8] # Magnetic field component in the z-direction
    
    # Contravariant time component of the four-velocity
    u⁰ = sqrt(u¹*u¹ + u²*u² + u³*u³ + 1)

    # Covariant components of the four-velocity
    u₀ = -u⁰ 
    u₁ =  u¹
    u₂ =  u²
    u₃ =  u³
    
    # Contravariant components of the magnetic four-vector
    b⁰ = B¹*u₁ + B²*u₂ + B³*u₃
    b¹ = (B¹ + b⁰*u¹)/u⁰
    b² = (B² + b⁰*u²)/u⁰
    b³ = (B³ + b⁰*u³)/u⁰	

    # Covariant components of the magnetic four-vector
    b₀ = -b⁰
    b₁ =  b¹
    b₂ =  b²
    b₃ =  b³	

    # Magnetic four-vector contraction            
    bsq = b⁰*b₀ + b¹*b₁ + b²*b₂ + b³*b₃

    #Total enthalpy
    enthalpy = ρ + eos.gamma*u + bsq
    
    # Conserved quantities in relativistic magnetohydrodynamics 
    U[1] = ρ*u⁰                                                  # Conserved mass density (D)
    U[2] = enthalpy*u⁰*u₀ + (eos.gamma - 1)*u + bsq/2 - b⁰*b₀    # Energy density (Q₀)
    U[3] = enthalpy*u⁰*u₁ - b⁰*b₁                                # Conserved momentum density in the x-direction (Q₁)    
    U[4] = enthalpy*u⁰*u₂ - b⁰*b₂                                # Conserved momentum density in the y-direction (Q₂)
    U[5] = enthalpy*u⁰*u₃ - b⁰*b₃                                # Conserved momentum density in the z-direction (Q₃)

end


@inline function function_PtoFx(P::AbstractVector{T}, F₁::AbstractVector{T}, eos::Polytrope{T}) where T<:Real

    # Primitive variables
    ρ  = P[1] # Rest-mass density
    u  = P[2] # Specific internal energy 
    u¹ = P[3] # Contravariant four-velocity in the x-direction
    u² = P[4] # Contravariant four-velocity in the y-direction
    u³ = P[5] # Contravariant four-velocity in the z-direction     
    B¹ = P[6] # Magnetic field component in the x-direction
    B² = P[7] # Magnetic field component in the y-direction
    B³ = P[8] # Magnetic field component in the z-direction
    
    # Contravariant time component of the four-velocity
    u⁰ = sqrt(u¹*u¹ + u²*u² + u³*u³ + 1)

    # Covariant components of the four-velocity
    u₀ = -u⁰ 
    u₁ =  u¹
    u₂ =  u²
    u₃ =  u³
    
    # Contravariant components of the magnetic four-vector
    b⁰ = B¹*u₁ + B²*u₂ + B³*u₃
    b¹ = (B¹ + b⁰*u¹)/u⁰
    b² = (B² + b⁰*u²)/u⁰
    b³ = (B³ + b⁰*u³)/u⁰	

    # Covariant components of the magnetic four-vector
    b₀ = -b⁰
    b₁ =  b¹
    b₂ =  b²
    b₃ =  b³	

    # Magnetic four-vector contraction            
    bsq = b⁰*b₀ + b¹*b₁ + b²*b₂ + b³*b₃

    #Total enthalpy
    enthalpy = ρ + eos.gamma*u + bsq

    # Fluxes in x-direction
    F₂[1] = ρ*u¹
    F₂[2] = enthalpy*u¹*u₀ - b¹*b₀
    F₂[3] = enthalpy*u¹*u₁ + (eos.gamma - 1)*u + bsq/2 - b¹*b₁
    F₂[4] = enthalpy*u¹*u₂ - b¹*b₂
    F₂[5] = enthalpy*u¹*u₃ - b¹*b₃
end



@inline function function_PtoFy(P::AbstractArray{T}, F₂::AbstractArray{T}, eos::Polytrope{T}) where T<:Real
    
    # Primitive variables
    ρ  = P[1] # Rest-mass density
    u  = P[2] # Specific internal energy 
    u¹ = P[3] # Contravariant four-velocity in the x-direction
    u² = P[4] # Contravariant four-velocity in the y-direction
    u³ = P[5] # Contravariant four-velocity in the z-direction     
    B¹ = P[6] # Magnetic field component in the x-direction
    B² = P[7] # Magnetic field component in the y-direction
    B³ = P[8] # Magnetic field component in the z-direction
    
    # Contravariant time component of the four-velocity
    u⁰ = sqrt(u¹*u¹ + u²*u² + u³*u³ + 1)

    # Covariant components of the four-velocity
    u₀ = -u⁰ 
    u₁ =  u¹
    u₂ =  u²
    u₃ =  u³
    
    # Contravariant components of the magnetic four-vector
    b⁰ = B¹*u₁ + B²*u₂ + B³*u₃
    b¹ = (B¹ + b⁰*u¹)/u⁰
    b² = (B² + b⁰*u²)/u⁰
    b³ = (B³ + b⁰*u³)/u⁰	

    # Covariant components of the magnetic four-vector
    b₀ = -b⁰
    b₁ =  b¹
    b₂ =  b²
    b₃ =  b³	

    # Magnetic four-vector contraction            
    bsq = b⁰*b₀ + b¹*b₁ + b²*b₂ + b³*b₃

    #Total enthalpy
    enthalpy = ρ + eos.gamma*u + bsq

    # Fluxes in y-direction
    F₂[1] = ρ*u²
    F₂[2] = enthalpy*u²*u₀ - b²*b₀
    F₂[3] = enthalpy*u²*u₁ - b²*b₁
    F₂[4] = enthalpy*u²*u₂ + (eos.gamma - 1)*u + bsq/2 - b²*b₂
    F₂[5] = enthalpy*u²*u₃ - b²*b₃
end


@inline function function_PtoFz(P::AbstractArray{T}, F₃::AbstractArray{T},eos::Polytrope{T}) where T<:Real

    # Primitive variables
    ρ  = P[1] # Rest-mass density
    u  = P[2] # Specific internal energy 
    u¹ = P[3] # Contravariant four-velocity in the x-direction
    u² = P[4] # Contravariant four-velocity in the y-direction
    u³ = P[5] # Contravariant four-velocity in the z-direction     
    B¹ = P[6] # Magnetic field component in the x-direction
    B² = P[7] # Magnetic field component in the y-direction
    B³ = P[8] # Magnetic field component in the z-direction
    
    # Contravariant time component of the four-velocity
    u⁰ = sqrt(u¹*u¹ + u²*u² + u³*u³ + 1)

    # Covariant components of the four-velocity
    u₀ = -u⁰ 
    u₁ =  u¹
    u₂ =  u²
    u₃ =  u³
    
    # Contravariant components of the magnetic four-vector
    b⁰ = B¹*u₁ + B²*u₂ + B³*u₃
    b¹ = (B¹ + b⁰*u¹)/u⁰
    b² = (B² + b⁰*u²)/u⁰
    b³ = (B³ + b⁰*u³)/u⁰	

    # Covariant components of the magnetic four-vector
    b₀ = -b⁰
    b₁ =  b¹
    b₂ =  b²
    b₃ =  b³	

    # Magnetic four-vector contraction            
    bsq = b⁰*b₀ + b¹*b₁ + b²*b₂ + b³*b₃

    #Total enthalpy
    enthalpy = ρ + eos.gamma*u + bsq

    # Fluxes in y-direction
    F₃[1] = ρ*u³
    F₃[2] = enthalpy*u³*u₀ - b³*b₀
    F₃[3] = enthalpy*u³*u₁ - b³*b₁
    F₃[4] = enthalpy*u³*u₂ - b³*b₂
    F₃[5] = enthalpy*u³*u₃ + (eos.gamma - 1)*u + bsq/2 - b³*b₃
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



@kernel inbounds = true function function_UtoP(@Const(U::AbstractArray{T}), P::AbstractArray{T}, P_stagger::AbstractArray{T}, eos::Polytrope{T},n_iter::Int64,tol::T=1e-10) where T<:Real
    
    i, j, k = @index(Global, NTuple)
    il, jl, kl = @index(Local, NTuple)

    @uniform begin
       N,M,L = @groupsize()
       Nx,Ny,Nz = @ndrange()
    end
    
    P_loc = @localmem eltype(U) (5,N,M,L)
    U_loc = @localmem eltype(U) (5,N,M,L)
    B_loc = @localmem eltype(P_stagger) (3,N,M,L)

    
    for idx in 1:5
       P_loc[idx,il,jl,kl] = P[idx,i,j,k]
       U_loc[idx,il,jl,kl] = U[idx,i,j,k]
    end

    B_loc[1,il,jl,kl] = 1/2 *(P_stagger[1,i+1,j,k] + P_stagger[1,i,j,k])
    B_loc[2,il,jl,kl] = 1/2 *(P_stagger[2,i,j+1,k] + P_stagger[2,i,j,k]) 
    B_loc[3,il,jl,kl] = 1/2 *(P_stagger[3,i,j,k+1] + P_stagger[3,i,j,k])

    buff_out_t_2D = @localmem eltype(U) (2,N,M,L)
    buff_out_2D   = @view buff_out_t_2D[:,il,jl,kl]
    
    buff_fun_t_2D = @localmem eltype(U) (2,N,M,L)
    buff_fun_2D   = @view buff_fun_t_2D[:,il,jl,kl]
    buff_jac_2D   = @MVector zeros(T,4)

    if i > 3 && i < Nx - 3 && j > 3 && j < Ny-3 && k > 3 && k < Nz-3

        #Conserved Variable
        D  = U_loc[1,il,jl,kl] # Conserved mass density 
        Q₀ = U_loc[2,il,jl,kl] # Conserved energy density 
        Q₁ = U_loc[3,il,jl,kl] # Conserved momentum density in the x-direction
        Q₂ = U_loc[4,il,jl,kl] # Conserved momentum density in the x-direction
        Q₃ = U_loc[5,il,jl,kl] # Conserved momentum density in the x-direction
        B¹ = B_loc[1,il,jl,kl] # Magnetic field component in the x-direction
        B² = B_loc[2,il,jl,kl] # Magnetic field component in the y-direction
        B³ = B_loc[3,il,jl,kl] # Magnetic field component in the z-direction

        #Useful Values
        S²  = Q₁*Q₁ + Q₂*Q₂ + Q₃*Q₃
        BSQ = B¹*B¹ + B²*B² + B³*B³
        QᵢBⁱ   = Q₁*B¹ + Q₂*B² + Q₃*B³
        α   = (eos.gamma - 1)/eos.gamma
        
        #Convergence indicators
        convergence_1DW       = false
        convergence_2D        = false
        convergence_Dekker    = false

        #Epsilon value
        ε = 1e-14

        #Guess primitive variables
        ρ_guess  = P_loc[1,il,jl,kl] #Density
        u_guess  = P_loc[2,il,jl,kl] #Internal Energy 
        u¹_guess = P_loc[3,il,jl,kl] #Contravariant Four-velocity in x-direction
        u²_guess = P_loc[4,il,jl,kl] #Contravariant Four-velocity in y-direction
        u³_guess = P_loc[5,il,jl,kl] #Contravariant Four-velocity in z-direction 
        
        #Guess useful values
        γ_guess  = sqrt(u¹_guess*u¹_guess + u²_guess*u²_guess + u³_guess*u³_guess + 1)
        w_guess  = ρ_guess + eos.gamma * u_guess
        v²_guess = (u¹_guess*u¹_guess + u²_guess*u²_guess + u³_guess*u³_guess) / γ_guess^2
        
        # v² less than c² and larger than 0
        if v²_guess >= 1                                      
            v²_guess = 1 - ε
        elseif v²_guess < 0
            v²_guess = ε
        end

        #1DW METHOD (Noble et al. 2006)
        if !convergence_1DW 

            # Boundary for W
            W = w_guess * γ_guess^2
            W_min = sqrt(S²) * (1 + ε) #Approximately
            W_max = 1e30     
   
            # Initial condition, with v²<1
            while (S²*W^2 + (QᵢBⁱ)^2 *(BSQ +2*W))/((BSQ + W)^2 *W^2) >= 1 && W < W_max
                W *= 10
            end
            
            #Main loop 1DW Newton–Raphson method
            for _ in 1:n_iter
                
                # W should be greater than W_min
                if W < W_min
                    W = W_min       
                end

                # Function for v²
                v² = (S²*W^2 + (QᵢBⁱ)^2 *(BSQ +2*W))/((BSQ + W)^2 *W^2)

                # v² less than c² and larger than 0             
                if v² < 0.0
                    v² = ε
                elseif v² > 1 -  ε
                    v² = 1 -  ε
                end	        
                
                # Main function + jacobian
                buff_fun = Q₀ + BSQ/2 *(1 + v²) -((QᵢBⁱ)^2 /(2*W^2)) + W - α*(W*(1 - v²) - D*sqrt(1 - v²))
                d_v²_dW  = -2 * ((BSQ)^2 * (QᵢBⁱ)^2 + 3 * BSQ * (QᵢBⁱ)^2 * W + 3 * (QᵢBⁱ)^2 * W^2 + S² * W^3)/(W^3 * (BSQ + W)^3)
                buff_jac = BSQ/2 *d_v²_dW  + ((QᵢBⁱ)^2 /(W^3)) + 1 - α * ((1 - v²) - W * d_v²_dW + D * d_v²_dW / (2 * sqrt(1 - v²)))                
                
                #Newton–Raphson method evolution 
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

                #Convergence condition
                if ΔW^2 < tol^2   
                    convergence_1DW = true
                    break
                end
		
		    end
        end
        
        #2D METHOD (Noble et al. 2006)
        if !convergence_1DW             
            
            #W and v² guess from previous step
            W  = w_guess * γ_guess^2
            v² = (u¹_guess*u¹_guess + u²_guess*u²_guess + u³_guess*u³_guess) / γ_guess^2

            #Main loop 2D Newton–Raphson method
            for _ in 1:n_iter 
                W_old  = W
                v²_old = v²

                buff_fun_2D[1] = v²*(BSQ+W)^2 - S² - (QᵢBⁱ)^2 *(BSQ+2*W)/(W^2)
                buff_fun_2D[2] = Q₀ + W + BSQ/2 *(1+v²) -(QᵢBⁱ)^2 /(2*W^2) - α*(W*(1 - v²) - D*sqrt(max(0.0,1 - v²)))

                buff_jac_2D[1] = 2*v²*(BSQ+W) + 2*(QᵢBⁱ)^2 * (BSQ+W)/W^3 
                buff_jac_2D[2] = 1 + (QᵢBⁱ)^2 / W^3 - α*(1 - v²)
                buff_jac_2D[3] = (BSQ+W)^2 
                buff_jac_2D[4] = BSQ/2 + α*W - α*D/(2*sqrt(max(0.0,1 - v²)))
                
                LU_dec_2D!(buff_jac_2D, buff_fun_2D, buff_out_2D)
                
                ΔW  = buff_out_2D[1]
                Δv² = buff_out_2D[2]
                
                W  = W  - ΔW
                v² = v² - Δv²        
                
                if ΔW^2 + Δv²^2 < tol^2
                    convergence_2D = true
                    break
                end                
            end
        end

        
        #1DW METHOD with Dekker Method (Noble et al. 2006)
        if !convergence_1DW && !convergence_2D 
            
            #Starting Values
            W_min = max(ε, sqrt(S²))
            W_max = W_min * 10.0

            v²_min = (S²*W_min^2 + (QᵢBⁱ)^2 *(BSQ +2*W_min))/((BSQ + W_min)^2 *W_min^2)
            v²_max = (S²*W_max^2 + (QᵢBⁱ)^2 *(BSQ +2*W_max))/((BSQ + W_max)^2 *W_max^2)

            fun_min = Q₀ + BSQ/2 *(1 + v²_min) -((QᵢBⁱ)^2 /(2*W_min^2)) + W_min - α*(W_min*(1 - v²_min) - D*sqrt(max(0.0,1 - v²_min)))
            fun_max = Q₀ + BSQ/2 *(1 + v²_max) -((QᵢBⁱ)^2 /(2*W_max^2)) + W_max - α*(W_max*(1 - v²_max) - D*sqrt(max(0.0,1 - v²_max)))

            #Looking for the root
            count = 0
            while fun_min*fun_max > 0 && count < 1000
                W_max  *= 10.0
                v²_max = (S²*W_max^2 + (QᵢBⁱ)^2 *(BSQ +2*W_max))/((BSQ + W_max)^2 *W_max^2)
                fun_max = Q₀ + BSQ/2 *(1 + v²_max) -((QᵢBⁱ)^2 /(2*W_max^2)) + W_max - α*(W_max*(1 - v²_max) - D*sqrt(max(0.0,1 - v²_max)))
                count += 1
            end
            
            
            if fun_min * fun_max <= 0
                for i in 1:n_iter
                    W_sec = W_max - fun_max*(W_max - W_min)/(fun_max - fun_min + ε)

                    if W_sec > W_min && W_sec < W_max
                        W_trial = W_sec
                    else
                        W_trial = 0.5*(W_min + W_max)
                    end

                    v²_trial  = (S²*W_trial^2 + (QᵢBⁱ)^2*(BSQ + 2*W_trial))/((BSQ + W_trial)^2 * W_trial^2)
                    fun_trial = Q₀ + BSQ/2*(1 + v²_trial) - (QᵢBⁱ)^2/(2*W_trial^2) + W_trial - α*(W_trial*(1 - v²_trial) - D*sqrt(max(0.0,1 - v²_trial)))

                    if fun_trial^2 < tol^2
                        W = W_trial
                        convergence_Dekker = true
                        break
                    end

                    if fun_min * fun_trial < 0
                        W_max   = W_trial
                        fun_max = fun_trial
                    else
                        W_min   = W_trial
                        fun_min = fun_trial
                    end

                    W = W_trial
                end
            else
                # Be careful
                # W = W_max
                # convergence_Dekker = true
            end
        end
        
        if convergence_1DW || convergence_2D || convergence_Dekker
            v² = (S²*W^2 + (QᵢBⁱ)^2 *(BSQ +2*W))/((BSQ + W)^2 *W^2)
            γ  = min(1/sqrt(1-v²), 50)  	             #LORENTZ FACTOR LIMITER
            ρ  = max(1e-8, D/γ) 		             #DENSITY FLOOR 
            u  = max(1e-8, (W / γ^2 - D / γ) / eos.gamma)    #INTERNAL ENERGY FLOOR
            u¹ = γ/(W+BSQ) * (Q₁+ QᵢBⁱ*B¹/W)
            u² = γ/(W+BSQ) * (Q₂+ QᵢBⁱ*B²/W)
            u³ = γ/(W+BSQ) * (Q₃+ QᵢBⁱ*B³/W)
            
            ########################################
            #FLOOR for the magnetization for ρ and u
            ########################################
            
            #Contravariant Four-magnetic field
            b⁰ = B¹*u₁ + B²*u₂ + B³*u₃
            b¹ = (B¹ + b⁰*u¹)/u⁰
            b² = (B² + b⁰*u²)/u⁰
            b³ = (B³ + b⁰*u³)/u⁰
        
            #Contravariant Four-magnetic field
            b₀ = -b⁰
            b₁ =  b¹
            b₂ =  b²
            b₃ =  b³
            
            #Useful Values            
            bsq = b⁰*b₀ + b¹*b₁ + b²*b₂ + b³*b₃		
            
            if (bsq/ρ) > 50:
                ρ = bsq/50
            end

            if (bsq/u) > 100:
                u = bsq/100
            end            

            Ploc[1,il,jl,kl] = ρ
            Ploc[2,il,jl,kl] = u
            Ploc[3,il,jl,kl] = u¹
            Ploc[4,il,jl,kl] = u²
            Ploc[5,il,jl,kl] = u³
        end
        
        
        #STOP CODE, WHEN DOES NOT CONVERGE
        if !convergence_1DW && !convergence_2D && !convergence_Dekker
            Ploc[1,il,jl,kl] = sqrt(-1)
            Ploc[2,il,jl,kl] = sqrt(-1)
            Ploc[3,il,jl,kl] = sqrt(-1)
            Ploc[4,il,jl,kl] = sqrt(-1)
            Ploc[5,il,jl,kl] = sqrt(-1)
        end
                        
    end

    @synchronize
    
    for idx in 1:5
    	P[idx,i,j,k] = Ploc[idx,il,jl,kl]
    end
end
