#  ██╗   ██╗███████╗██████╗  ██████╗ ███╗   ██╗ █████╗ 
#  ██║   ██║██╔════╝██╔══██╗██╔═══██╗████╗  ██║██╔══██╗
#  ██║   ██║█████╗  ██████╔╝██║   ██║██╔██╗ ██║███████║
#  ╚██╗ ██╔╝██╔══╝  ██╔══██╗██║   ██║██║╚██╗██║██╔══██║
#   ╚████╔╝ ███████╗██║  ██║╚██████╔╝██║ ╚████║██║  ██║
#    ╚═══╝  ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝

#Magnetohydrodynamics in special relativity.

#Primitive variables:
#	ρ  - density 						- P[1,i,j,k]
#	u  - internal energy 					- P[2,i,j,k] 
#	u¹ - Contravariant Four-velocity in x-direction 	- P[3,i,j,k] 
#	u² - Contravariant Four-velocity in y-direction 	- P[4,i,j,k] 
#	u³ - Contravariant Four-velocity in z-direction 	- P[5,i,j,k]  
#	B¹ - Magnetic field in x-direction 			- P[6,i,j,k] 
#	B² - Magnetic field in y-direction 			- P[7,i,j,k]
#	B³ - Magnetic field in z-direction 			- P[8,i,j,k] 

using LinearAlgebra
using StaticArrays
using Base.Threads
using Base.Iterators
using KernelAbstractions
using CUDA
using MPI
using HDF5


function local_to_global(i,p,Size,MPI)
    if p == 0
        return i
    elseif p > 0 && p < MPI-1 && (p < 4 || px > Size-3) 
        return 0
    else
        return i + p * (Size - 6)
    end
end


abstract type FlowArr{T} end


mutable struct ParVector3D{T <:Real} <: FlowArr{T}
    # Parameter Vector
    arr::Array{T,4}
    size_X::Int64
    size_Y::Int64
    size_Z::Int64
    function ParVector3D{T}(Nx,Ny,Nz) where {T}
        arr = zeros(T, 5, Nx + 6, Ny + 6, Nz + 6)
        new(arr,Nx + 6 , Ny + 6, Nz + 6)
    end
    function ParVector3D{T}(arr::FlowArr{T}) where {T}
        new(Array{T}(arr.arr), arr.size_X, arr.size_Y, arr.size_Z)
    end
end

mutable struct CuParVector3D{T <:Real} <: FlowArr{T}
    # Parameter Vector
    arr::CuArray{T}
    size_X::Int64
    size_Y::Int64
    size_Z::Int64
    function CuParVector3D{T}(arr::FlowArr{T}) where {T}
        new(CuArray{T}(arr.arr), arr.size_X, arr.size_Y, arr.size_Z)
    end

    function CuParVector3D{T}(Nx::Int64, Ny::Int64, Nz::Int64) where {T}
        new(CuArray{T}(zeros(T, 5, Nx + 6, Ny + 6, Nz + 6)), Nx + 6, Ny + 6, Nz + 6)
    end
end

function VectorLike(X::FlowArr{T}) where T
    if typeof(X.arr) <: CuArray
        return CuParVector3D{T}(X.size_X - 6, X.size_Y - 6, X.size_Z - 6)
    else
        return ParVector3D{T}(X.size_X - 6, X.size_Y - 6, X.size_Z - 6)
    end
end

@kernel inbounds = true function kernel_PtoU(@Const(P::AbstractArray{T}), U::AbstractArray{T},eos::Polytrope{T}) where T
    i, j, k = @index(Global, NTuple)    
    begin

	#Primitive variables
	ρ  = P[1,i,j,k] #Density
	u  = P[2,i,j,k] #Internal Energy 
	u¹ = P[3,i,j,k] #Contravariant Four-velocity in x-direction
	u² = P[4,i,j,k] #Contravariant Four-velocity in y-direction
	u³ = P[5,i,j,k] #Contravariant Four-velocity in z-direction   
	B¹ = P[6,i,j,k] #Magnetic field in x-direction
	B² = P[7,i,j,k] #Magnetic field in y-direction
	B³ = P[8,i,j,k] #Magnetic field in z-direction
	
	#Contravariant Four-velocity in t-direction
	u⁰ = sqrt(u¹^2 + u²^2 + u³^2 + 1)
	
	#Covariant Four-velocities
	u₀ = -u⁰ 
	u₁ =  u¹
	u₂ =  u²
	u₃ =  u³	
	
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
	bsq    = b⁰*b₀ + b¹*b₁ + b²*b₂ + b³*b₃
	value  = ρ + eos.gamma*u + bsq 
	value2 = (eos.gamma - 1)*u + bsq/2 

        U[1,i,j,k] = ρ*u⁰
        U[2,i,j,k] = value*u⁰*u₀ + value2 - b⁰*b₀
        U[3,i,j,k] = value*u⁰*u₁ - b⁰*b₁
        U[4,i,j,k] = value*u⁰*u₂ - b⁰*b₂
        U[5,i,j,k] = value*u⁰*u₃ - b⁰*b₃    
        U[6,i,j,k] = B¹
        U[7,i,j,k] = B²
        U[8,i,j,k] = B³
    end
end


@inline function function_PtoU(P::AbstractVector{T}, U::AbstractVector{T},eos::Polytrope{T}) where T

	#Primitive variables
	ρ  = P[1] #Density
	u  = P[2] #Internal Energy 
	u¹ = P[3] #Contravariant Four-velocity in x-direction
	u² = P[4] #Contravariant Four-velocity in y-direction
	u³ = P[5] #Contravariant Four-velocity in z-direction   
	B¹ = P[6] #Magnetic field in x-direction
	B² = P[7] #Magnetic field in y-direction
	B³ = P[8] #Magnetic field in z-direction
	
	#Contravariant Four-velocity in t-direction
	u⁰ = sqrt(u¹^2 + u²^2 + u³^2 + 1)
	
	#Covariant Four-velocities
	u₀ = -u⁰ 
	u₁ =  u¹
	u₂ =  u²
	u₃ =  u³	
	
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
	bsq    = b⁰*b₀ + b¹*b₁ + b²*b₂ + b³*b₃
	value  = ρ + eos.gamma*u + bsq 
	value2 = (eos.gamma - 1)*u + bsq/2 

	#Conserved variables
        U[1] = ρ*u⁰
        U[2] = value*u⁰*u₀ + value2 - b⁰*b₀
        U[3] = value*u⁰*u₁ - b⁰*b₁
        U[4] = value*u⁰*u₂ - b⁰*b₂
        U[5] = value*u⁰*u₃ - b⁰*b₃    
        U[6] = B¹
        U[7] = B²
        U[8] = B³

end


@inline function function_PtoFx(P::AbstractVector{T}, Fx::AbstractVector{T},eos::Polytrope{T}) where T
	
	#Primitive variables
	ρ  = P[1] #Density
	u  = P[2] #Internal Energy 
	u¹ = P[3] #Contravariant Four-velocity in x-direction
	u² = P[4] #Contravariant Four-velocity in y-direction
	u³ = P[5] #Contravariant Four-velocity in z-direction   
	B¹ = P[6] #Magnetic field in x-direction
	B² = P[7] #Magnetic field in y-direction
	B³ = P[8] #Magnetic field in z-direction
	
	#Contravariant Four-velocity in t-direction
	u⁰ = sqrt(u¹^2 + u²^2 + u³^2 + 1)
	
	#Covariant Four-velocities
	u₀ = -u⁰ 
	u₁ =  u¹
	u₂ =  u²
	u₃ =  u³	
	
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
	bsq    = b⁰*b₀ + b¹*b₁ + b²*b₂ + b³*b₃
	value  = ρ + eos.gamma*u + bsq 
	value2 = (eos.gamma - 1)*u + bsq/2 

	#Fluxes
	Fx[1] = ρ*u¹
	Fx[2] = value*u¹*u₀ - b¹*b₀
	Fx[3] = value*u¹*u₁ + value2 - b¹*b₁
	Fx[4] = value*u¹*u₂ - b¹*b₂
	Fx[5] = value*u¹*u₃ - b¹*b₃

end



@inline function function_PtoFy(P::AbstractArray{T}, Fy::AbstractArray{T},eos::Polytrope{T}) where T
	
	#Primitive variables
	ρ  = P[1] #Density
	u  = P[2] #Internal Energy 
	u¹ = P[3] #Contravariant Four-velocity in x-direction
	u² = P[4] #Contravariant Four-velocity in y-direction
	u³ = P[5] #Contravariant Four-velocity in z-direction   
	B¹ = P[6] #Magnetic field in x-direction
	B² = P[7] #Magnetic field in y-direction
	B³ = P[8] #Magnetic field in z-direction
	
	#Contravariant Four-velocity in t-direction
	u⁰ = sqrt(u¹^2 + u²^2 + u³^2 + 1)
	
	#Covariant Four-velocities
	u₀ = -u⁰ 
	u₁ =  u¹
	u₂ =  u²
	u₃ =  u³	
	
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
	bsq    = b⁰*b₀ + b¹*b₁ + b²*b₂ + b³*b₃
	value  = ρ + eos.gamma*u + bsq 
	value2 = (eos.gamma - 1)*u + bsq/2 

	#Fluxes
	Fy[1] = ρ*u²
	Fy[2] = value*u²*u₀ - b²*b₀
	Fy[3] = value*u²*u₁ - b²*b₁
	Fy[4] = value*u²*u₂ + value2 - b²*b₂
	Fy[5] = value*u²*u₃ - b²*b₃
end


@inline function function_PtoFz(P::AbstractArray{T}, Fy::AbstractArray{T},eos::Polytrope{T}) where T
	
	#Primitive variables
	ρ  = P[1] #Density
	u  = P[2] #Internal Energy 
	u¹ = P[3] #Contravariant Four-velocity in x-direction
	u² = P[4] #Contravariant Four-velocity in y-direction
	u³ = P[5] #Contravariant Four-velocity in z-direction   
	B¹ = P[6] #Magnetic field in x-direction
	B² = P[7] #Magnetic field in y-direction
	B³ = P[8] #Magnetic field in z-direction
	
	#Contravariant Four-velocity in t-direction
	u⁰ = sqrt(u¹^2 + u²^2 + u³^2 + 1)
	
	#Covariant Four-velocities
	u₀ = -u⁰ 
	u₁ =  u¹
	u₂ =  u²
	u₃ =  u³	
	
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
	bsq    = b⁰*b₀ + b¹*b₁ + b²*b₂ + b³*b₃
	value  = ρ + eos.gamma*u + bsq 
	value2 = (eos.gamma - 1)*u + bsq/2 
	
	#Fluxes
	Fy[1] = ρ*u³
	Fy[2] = value*u³*u₀ - b³*b₀
	Fy[3] = value*u³*u₁ - b³*b₁
	Fy[4] = value*u³*u₂ - b³*b₂
	Fy[5] = value*u³*u₃ + value2 - b³*b₃
end


@inline function LU_dec!(flat_matrix::AbstractVector{T}, target::AbstractVector{T}, x::AbstractVector{T}) where T

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

@kernel inbounds = true function function_UtoP(@Const(U::AbstractArray{T}), P::AbstractArray{T},eos::Polytrope{T},n_iter::Int64,tol::T=1e-10) where T
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
		
		#Primitive variables
		ρ  = P[1,il,jl,kl] #Density
		u  = P[2,il,jl,kl] #Internal Energy 
		u¹ = P[3,il,jl,kl] #Contravariant Four-velocity in x-direction
		u² = P[4,il,jl,kl] #Contravariant Four-velocity in y-direction
		u³ = P[5,il,jl,kl] #Contravariant Four-velocity in z-direction   
		B¹ = P[6,il,jl,kl] #Magnetic field in x-direction
		B² = P[7,il,jl,kl] #Magnetic field in y-direction
		B³ = P[8,il,jl,kl] #Magnetic field in z-direction

		#Contravariant Four-velocity in t-direction
		u⁰ = sqrt(u¹^2 + u²^2 + u³^2 + 1)

		#Covariant Four-velocities
		u₀ = -u⁰ 
		u₁ =  u¹
		u₂ =  u²
		u₃ =  u³	

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
		bsq    = b⁰*b₀ + b¹*b₁ + b²*b₂ + b³*b₃
		value  = ρ + eos.gamma*u + bsq 
		value2 = (eos.gamma - 1)*u + bsq/2 
		
		#Derivatives:
		u⁰_x1 = 0
		u⁰_x2 = 0
		u⁰_x3 = u¹/u⁰
		u⁰_x4 = u²/u⁰	
		u⁰_x5 = u³/u⁰
		
		#For u¹
		u¹_x1  = 0
		u¹_x2  = 0
		u¹_x3  = 1
		u¹_x4  = 0
		u¹_x5  = 0	

		#For u²
		u²_x1  = 0
		u²_x2  = 0
		u²_x3  = 0
		u²_x4  = 1
		u²_x5  = 0	

		#For u³
		u³_x1  = 0
		u³_x2  = 0
		u³_x3  = 0
		u³_x4  = 0
		u³_x5  = 1

		#For u₀ 
		u₀_x1  = 0
		u₀_x2  = 0	
		u₀_x3  = -u⁰_x3
		u₀_x4  = -u⁰_x4
		u₀_x5  = -u⁰_x5

		#For u₁ 
		u₁_x1  = 0
		u₁_x2  = 0	
		u₁_x3  = 1
		u₁_x4  = 0
		u₁_x5  = 0

		#For u₂ 
		u₂_x1  = 0
		u₂_x2  = 0	
		u₂_x3  = 0
		u₂_x4  = 1
		u₂_x5  = 0

		#For u₃ 
		u₃_x1  = 0
		u₃_x2  = 0	
		u₃_x3  = 0
		u₃_x4  = 0
		u₃_x5  = 1

		#For b⁰ 
		b⁰_x1  = 0
		b⁰_x2  = 0
		b⁰_x3  = B¹*u₁_x3 + B²*u₂_x3 + B³*u₃_x3
		b⁰_x4  = B¹*u₁_x4 + B²*u₂_x4 + B³*u₃_x4	
		b⁰_x5  = B¹*u₁_x5 + B²*u₂_x5 + B³*u₃_x5	

		#For b¹	
		b¹_x1  = 0
		b¹_x2  = 0
		b¹_x3  = ((b⁰_x3*u¹ + u¹_x3*b⁰)*u⁰ - (B¹ + b⁰*u¹)*u⁰_x3)/(u⁰^2)
		b¹_x4  = ((b⁰_x4*u¹ + u¹_x4*b⁰)*u⁰ - (B¹ + b⁰*u¹)*u⁰_x4)/(u⁰^2)	
		b¹_x5  = ((b⁰_x5*u¹ + u¹_x5*b⁰)*u⁰ - (B¹ + b⁰*u¹)*u⁰_x5)/(u⁰^2)	

		#For b² 		
		b²_x1  = 0
		b²_x2  = 0
		b²_x3  = ((b⁰_x3*u² + u²_x3*b⁰)*u⁰ - (B² + b⁰*u²)*u⁰_x3)/(u⁰^2)
		b²_x4  = ((b⁰_x4*u² + u²_x4*b⁰)*u⁰ - (B² + b⁰*u²)*u⁰_x4)/(u⁰^2)	
		b²_x5  = ((b⁰_x5*u² + u²_x5*b⁰)*u⁰ - (B² + b⁰*u²)*u⁰_x5)/(u⁰^2)		

		#For b³ 		
		b³_x1  = 0
		b³_x2  = 0
		b³_x3  = ((b⁰_x3*u³ + u³_x3*b⁰)*u⁰ - (B³+b⁰*u³)*u⁰_x3)/(u⁰^2)
		b³_x4  = ((b⁰_x4*u³ + u³_x4*b⁰)*u⁰ - (B³+b⁰*u³)*u⁰_x4)/(u⁰^2)	
		b³_x5  = ((b⁰_x5*u³ + u³_x5*b⁰)*u⁰ - (B³+b⁰*u³)*u⁰_x5)/(u⁰^2)	

		#For b₀ 		
		b₀_x1  = 0
		b₀_x2  = 0
		b₀_x3  = -b⁰_x3 
		b₀_x4  = -b⁰_x4	
		b₀_x5  = -b⁰_x5

		#For b₁ 		
		b₁_x1  = 0
		b₁_x2  = 0
		b₁_x3  =  b¹_x3
		b₁_x4  =  b¹_x4
		b₁_x5  =  b¹_x5	

		#For b₂ 		
		b₂_x1  =  0
		b₂_x2  =  0
		b₂_x3  =  b²_x3
		b₂_x4  =  b²_x4
		b₂_x5  =  b²_x5	

		#For b₃ 		
		b₃_x1  = 0
		b₃_x2  = 0
		b₃_x3  = b³_x3
		b₃_x4  = b³_x4
		b₃_x5  = b³_x5

		#For bsq
		bsq_x1  = 0
		bsq_x2  = 0
		bsq_x3  = b⁰_x3*b₀ + b⁰*b₀_x3 + b¹_x3*b₁ + b¹*b₁_x3 + b²_x3*b₂ + b²*b₂_x3 + b³_x3*b₃ + b³*b₃_x3
		bsq_x4  = b⁰_x4*b₀ + b⁰*b₀_x4 + b¹_x4*b₁ + b¹*b₁_x4 + b²_x4*b₂ + b²*b₂_x4 + b³_x4*b₃ + b³*b₃_x4	
		bsq_x5  = b⁰_x5*b₀ + b⁰*b₀_x5 + b¹_x5*b₁ + b¹*b₁_x5 + b²_x5*b₂ + b²*b₂_x5 + b³_x5*b₃ + b³*b₃_x5	

		#For value
		value_x1  = 1
		value_x2  = eos.gamma	
		value_x3  = bsq_x3	
		value_x4  = bsq_x4
		value_x5  = bsq_x5

		#For value2
		value2_x1  = 0
		value2_x2  = eos.gamma - 1	
		value2_x3  = (0.5)*bsq_x3	
		value2_x4  = (0.5)*bsq_x4
		value2_x5  = (0.5)*bsq_x5
		
		
		buff_fun[1] = ρ*u⁰                         - Uloc[1,il,jl,kl]
		buff_fun[2] = value*u⁰*u₀ + value2 - b⁰*b₀ - Uloc[2,il,jl,kl]
		buff_fun[3] = value*u⁰*u₁ - b⁰*b₁          - Uloc[3,il,jl,kl]
		buff_fun[4] = value*u⁰*u₂ - b⁰*b₂          - Uloc[4,il,jl,kl]
		buff_fun[5] = value*u⁰*u₃ - b⁰*b₃          - Uloc[5,il,jl,kl]            
		buff_fun[6] = B¹                           - Uloc[6,il,jl,kl]            
		buff_fun[7] = B²                           - Uloc[7,il,jl,kl]            
		buff_fun[8] = B³                           - Uloc[8,il,jl,kl]            


		buff_jac[1]  = u⁰
		buff_jac[6]  = 0   
		buff_jac[11] = ρ*u⁰_x3
		buff_jac[16] = ρ*u⁰_x4
		buff_jac[21] = ρ*u⁰_x5
 
		buff_jac[2]  =  value_x1*u⁰*u₀ + value2_x1
		buff_jac[7]  =  value_x2*u⁰*u₀ + value2_x2
		buff_jac[12] = (value_x3*(u⁰*u₀) + value*(u⁰_x3*u₀ + u₀_x3*u⁰)) + value2_x3 - (b⁰_x3*b₀ + b₀_x3*b⁰)
		buff_jac[17] = (value_x4*(u⁰*u₀)  + value*(u⁰_x4*u₀ + u₀_x4*u⁰)) + value2_x4 - (b⁰_x4*b₀ + b₀_x4*b⁰)
		buff_jac[22] = (value_x5*(u⁰*u₀) + value*(u⁰_x5*u₀ + u₀_x5*u⁰)) + value2_x5 - (b⁰_x5*b₀ + b₀_x5*b⁰)

		buff_jac[3]  =  value_x1*u⁰*u₁ 
		buff_jac[8]  =  value_x2*u⁰*u₁
		buff_jac[13] = (value_x3*(u⁰*u₁) + value*(u⁰_x3*u₁ + u₁_x3*u⁰)) - (b⁰_x3*b₁ + b₁_x3*b⁰)
		buff_jac[18] = (value_x4*(u⁰*u₁) + value*(u⁰_x4*u₁ + u₁_x4*u⁰)) - (b⁰_x4*b₁ + b₁_x4*b⁰)
		buff_jac[23] = (value_x5*(u⁰*u₁) + value*(u⁰_x5*u₁ + u₁_x5*u⁰)) - (b⁰_x5*b₁ + b₁_x5*b⁰)         

		buff_jac[4]  = value_x1*u⁰*u₂
		buff_jac[9]  = value_x2*u⁰*u₂
		buff_jac[14] = (value_x3*(u⁰*u₂) + value*(u⁰_x3*u₂ + u₂_x3*u⁰)) - (b⁰_x3*b₂ + b₂_x3*b⁰)
		buff_jac[19] = (value_x4*(u⁰*u₂) + value*(u⁰_x4*u₂ + u₂_x4*u⁰)) - (b⁰_x4*b₂ + b₂_x4*b⁰)
		buff_jac[24] = (value_x5*(u⁰*u₂) + value*(u⁰_x5*u₂ + u₂_x5*u⁰)) - (b⁰_x5*b₂ + b₂_x5*b⁰)       

		buff_jac[5]  = value_x1*u⁰*u₃
		buff_jac[10] = value_x2*u⁰*u₃
		buff_jac[15] = (value_x3*(u⁰*u₃) + value*(u⁰_x3*u₃ + u₃_x3*u⁰)) - (b⁰_x3*b₃ + b₃_x3*b⁰)
		buff_jac[20] = (value_x4*(u⁰*u₃) + value*(u⁰_x4*u₃ + u₃_x4*u⁰)) - (b⁰_x4*b₃ + b₃_x4*b⁰)
		buff_jac[25] = (value_x5*(u⁰*u₃) + value*(u⁰_x5*u₃ + u₃_x5*u⁰)) - (b⁰_x5*b₃ + b₃_x5*b⁰)
            
            
            LU_dec!(buff_jac,buff_fun,buff_out)

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
