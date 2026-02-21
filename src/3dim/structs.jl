using LinearAlgebra
using StaticArrays
using Base.Threads
using Base.Iterators
using KernelAbstractions
using CUDA
using MPI
using HDF5

abstract type VeronaArr{T} end


mutable struct ParVector3D{T<:Real} <: VeronaArr{T}
    # Parameter Vector
    arr::Array{T,4}
    size_X::Int64
    size_Y::Int64
    size_Z::Int64
    function ParVector3D{T}(Nx, Ny, Nz) where {T}
        arr = zeros(T, 5, Nx + 6, Ny + 6, Nz + 6)
        new(arr, Nx + 6, Ny + 6, Nz + 6)
    end
    function ParVector3D{T}(arr::VeronaArr{T}) where {T}
        new(Array{T}(arr.arr), arr.size_X, arr.size_Y, arr.size_Z)
    end
end

mutable struct CuParVector3D{T<:Real} <: VeronaArr{T}
    # Parameter Vector
    arr::CuArray{T}
    size_X::Int64
    size_Y::Int64
    size_Z::Int64
    function CuParVector3D{T}(arr::VeronaArr{T}) where {T<:Real}
        new(CuArray{T}(arr.arr), arr.size_X, arr.size_Y, arr.size_Z)
    end

    function CuParVector3D{T}(Nx::Int64, Ny::Int64, Nz::Int64) where {T<:Real}
        new(CuArray{T}(zeros(T, 5, Nx + 6, Ny + 6, Nz + 6)), Nx + 6, Ny + 6, Nz + 6)
    end
end

function VectorLike(X::VeronaArr{T}) where {T}
    if typeof(X.arr) <: CuArray
        return CuParVector3D{T}(X.size_X - 6, X.size_Y - 6, X.size_Z - 6)
    else
        return ParVector3D{T}(X.size_X - 6, X.size_Y - 6, X.size_Z - 6)
    end
end

@kernel inbounds = true function kernel_PtoU(
    @Const(P::AbstractArray{T}),
    U::AbstractArray{T},
    eos::Polytrope{T},
) where {T<:Real}
    i, j, k = @index(Global, NTuple)
    begin

		# Primitive variables
		ρ  = P[1,i,j,k] 		    # Rest-mass density
		v¹ = P[2,i,j,k] 	        # Contravariant three-velocity in the x-direction
		v² = P[3,i,j,k] 		    # Contravariant three-velocity in the y-direction
		v³ = P[4,i,j,k] 		    # Contravariant three-velocity in the z-direction     
		u  = P[5,i,j,k] 	        # Specific internal energy 

		# Useful
		W = 1/sqrt(1-(v¹*v¹ + v²*v² + v³*v³)) #Lorentz factor
		h = 1 + (eos.gamma)*u                 #Enthalpy

		#Conserved variables        
        U[1, i, j, k] =  ρ*W 
        U[2, i, j, k] = (ρ*h)*W^2 * v¹ 
        U[3, i, j, k] = (ρ*h)*W^2 * v²  
        U[4, i, j, k] = (ρ*h)*W^2 * v³ 
        U[5, i, j, k] = (ρ*h)*W^2 - ρ*u*(eos.gamma -1) - ρ*W 
    end
end


@inline function function_PtoU(
    P::AbstractVector{T},
    U::AbstractVector{T},
    eos::Polytrope{T},
) where {T<:Real}
	
    # Primitive variables
    ρ  = P[1] 		    # Rest-mass density
    v¹ = P[2] 	        # Contravariant three-velocity in the x-direction
    v² = P[3] 		    # Contravariant three-velocity in the y-direction
    v³ = P[4] 		    # Contravariant three-velocity in the z-direction     
    u  = P[5] 	        # Specific internal energy 
	
    # Useful
	W = 1/sqrt(1-(v¹*v¹ + v²*v² + v³*v³))
	h = 1 + (eos.gamma)*u
    
	#Conserved variables
	U[1] = ρ*W 
	U[2] = ρ*h*W^2 * v¹       
	U[3] = ρ*h*W^2 * v²            
	U[4] = ρ*h*W^2 * v³ 
	U[5] = ρ*h*W^2 - ρ*u*(eos.gamma-1) - ρ*W 
end


@inline function function_PtoFx(
    P::AbstractVector{T},
    Fx::AbstractVector{T},
    eos::Polytrope{T},
) where {T<:Real}

    # Primitive variables
    ρ  = P[1] 		    # Rest-mass density
    v¹ = P[2] 	        # Contravariant three-velocity in the x-direction
    v² = P[3] 		    # Contravariant three-velocity in the y-direction
    v³ = P[4] 		    # Contravariant three-velocity in the z-direction     
    u  = P[5] 	        # Specific internal energy 
    
    # Useful
    W = 1/sqrt(1-(v¹*v¹ + v²*v² + v³*v³))
    h = 1 + (eos.gamma)*u
    
    #Conserved variables
    D  = ρ*W
    S₁ = ρ*h*W^2 * v¹
    S₂ = ρ*h*W^2 * v²
    S₃ = ρ*h*W^2 * v³ 
    τ  = ρ*h*W^2 - ρ*u*(eos.gamma -1) - D	
	
    #Fluxes in X-direction
    Fx[1] = D *v¹
    Fx[2] = S₁*v¹ + ρ*u*(eos.gamma -1) 
    Fx[3] = S₂*v¹              
    Fx[4] = S₃*v¹  
    Fx[5] = τ *v¹ + ρ*u*(eos.gamma -1)*v¹ 
end



@inline function function_PtoFy(
    P::AbstractArray{T},
    Fy::AbstractArray{T},
    eos::Polytrope{T},
) where {T<:Real}
    # Primitive variables
    ρ  = P[1] 		    # Rest-mass density
    v¹ = P[2] 	        # Contravariant three-velocity in the x-direction
    v² = P[3] 		    # Contravariant three-velocity in the y-direction
    v³ = P[4] 		    # Contravariant three-velocity in the z-direction     
    u  = P[5] 	        # Specific internal energy 
    
    # Useful
    W = 1/sqrt(1-(v¹*v¹ + v²*v² + v³*v³))
    h = 1 + (eos.gamma)*u
    
    #Conserved variables
    D  = ρ*W
    S₁ = ρ*h*W^2 * v¹
    S₂ = ρ*h*W^2 * v²
    S₃ = ρ*h*W^2 * v³ 
    τ  = ρ*h*W^2 - ρ*u*(eos.gamma -1) - D	

    #Fluxes in Y-direction
    Fy[1] = D *v²
    Fy[2] = S₁*v² 
    Fy[3] = S₂*v² + ρ*u*(eos.gamma -1)           
    Fy[4] = S₃*v² 
    Fy[5] = τ *v² + ρ*u*(eos.gamma -1)*v²
end


@inline function function_PtoFz(
    P::AbstractArray{T},
    Fz::AbstractArray{T},
    eos::Polytrope{T},
) where {T<:Real}
    
# Primitive variables
    ρ  = P[1] 		    # Rest-mass density
    v¹ = P[2] 	        # Contravariant three-velocity in the x-direction
    v² = P[3] 		    # Contravariant three-velocity in the y-direction
    v³ = P[4] 		    # Contravariant three-velocity in the z-direction     
    u  = P[5] 	        # Specific internal energy 
    
    # Useful
    W = 1/sqrt(1-(v¹*v¹ + v²*v² + v³*v³))
    h = 1 + (eos.gamma)*u
    
    #Conserved variables
    D  = ρ*W
    S₁ = ρ*h*W^2 * v¹
    S₂ = ρ*h*W^2 * v²
    S₃ = ρ*h*W^2 * v³ 
    τ  = ρ*h*W^2 - ρ*u*(eos.gamma -1) - D
    
    #Fluxes in Z-direction
    Fz[1] = D *v³
    Fz[2] = S₁*v³
    Fz[3] = S₂*v³             
    Fz[4] = S₃*v³ + ρ*u*(eos.gamma -1)
    Fz[5] = τ *v³ + ρ*u*(eos.gamma -1)*v³
end

@kernel inbounds = true function function_UtoP(
    @Const(U::AbstractArray{T}),
    P::AbstractArray{T},
    eos::Polytrope{T},
    n_iter::Int64,
    tol::T = 1e-10,
) where {T<:Real}

    i, j, k = @index(Global, NTuple)
    il, jl, kl = @index(Local, NTuple)

    @uniform begin
        N, M, L = @groupsize()
        Nx, Ny, Nz = @ndrange()
    end

    Ploc = @localmem eltype(U) (5, N, M, L)
    Uloc = @localmem eltype(U) (5, N, M, L)

    for idx = 1:5
        Ploc[idx, il, jl, kl] = P[idx, i, j, k]
        Uloc[idx, il, jl, kl] = U[idx, i, j, k]
    end

    if i > 3 && i < Nx - 2 && j > 3 && j < Ny-2 && k > 3 && k < Nz-2

        #Conserved Variable
        D  = Uloc[1, il, jl, kl]
        S₁ = Uloc[2, il, jl, kl]
        S₂ = Uloc[3, il, jl, kl]
        S₃ = Uloc[4, il, jl, kl]
        τ  = Uloc[5, il, jl, kl]

        #Useful
        S² = S₁*S₁ + S₂*S₂ + S₃*S₃
        Z_min = max(D, sqrt(S²)*(1+1e-12))
	    Z_max = max(Z_min*2, D + eos.gamma*τ + 10*abs(τ))
        a = Z_min
        b = Z_max     
        
        #Function for BRENT solver
        f(z) = begin
		τ + D - z + ((eos.gamma - 1)/eos.gamma) * ( z*(1 - (S² / z^2)) - D*sqrt(1 -  ((S² / z^2))))
        end

        #BRENT SOLVER
        fa, fb = f(a), f(b)
        if abs(fa) < abs(fb)
            a, b = b, a
            fa, fb = fb, fa
        end
        c, fc, d = a, fa, 0.0
        mflag = true
        tolerance = 1e-8
        converged = false

        for _ in 1:100
            if fb == 0 || abs(b - a) <= tolerance
                converged = true
                break
            end

            s = if fa != fc && fb != fc
                (a*fb*fc)/((fa - fb)*(fa - fc)) +
                (b*fa*fc)/((fb - fa)*(fb - fc)) +
                (c*fa*fb)/((fc - fa)*(fc - fb))
            else
                b - fb*(b - a)/(fb - fa)
            end

            if ((s - (3*a + b)/4)*(s - b) >= 0) || 
               (mflag && abs(s - b) >= abs(b - c)/2) ||
               (!mflag && abs(s - b) >= abs(c - d)/2) ||
               (mflag && abs(b - c) < tolerance) ||
               (!mflag && abs(c - d) < tolerance)
                s = (a + b)/2
                mflag = true
            else
                mflag = false
            end

            d, c, fc = c, b, fb
            fs = f(s)

            if fa * fs < 0
                b, fb = s, fs
            else
                a, fa = s, fs
            end

            if abs(fa) < abs(fb)
                a, b = b, a
                fa, fb = fb, fa
            end
        end

        Z_SOL = b
        vsq = S²/(Z_SOL)^2
        W = 1/sqrt(1 - vsq)
        if converged
            Ploc[1, il, jl, kl] = D/W
            Ploc[2, il, jl, kl] = S₁/(Z_SOL)
            Ploc[3, il, jl, kl] = S₂/(Z_SOL)
            Ploc[4, il, jl, kl] = S₃/(Z_SOL)
            Ploc[5, il, jl, kl] = 1/eos.gamma *(Z_SOL/((D/W)*W^2)-1)
        else
            Ploc[1, il, jl, kl] = sqrt(-1)
            Ploc[2, il, jl, kl] = sqrt(-1)
            Ploc[3, il, jl, kl] = sqrt(-1)
            Ploc[4, il, jl, kl] = sqrt(-1)
            Ploc[5, il, jl, kl] = sqrt(-1)
        end       
    end

    @synchronize

    for idx = 1:5
        P[idx, i, j, k] = Ploc[idx, il, jl, kl]
    end
end
