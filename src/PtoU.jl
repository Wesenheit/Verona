@inline function function_PtoU_kernel!(P::AbstractArray{T}, B1::AbstractArray{T}, B2::AbstractArray{T}, B3::AbstractArray{T}, U::AbstractArray{T}, i, j, k) where T<:Real
    @inbounds begin
        # Primitive variables
        ρ  = P[1,i,j,k]                                       # Rest-mass density
	u  = P[2,i,j,k]                                       # Specific internal energy 
        u¹ = P[3,i,j,k]                                       # Contravariant four-velocity in the x-direction
        u² = P[4,i,j,k]                                       # Contravariant four-velocity in the y-direction
        u³ = P[5,i,j,k]                                       # Contravariant four-velocity in the z-direction     
        B¹ = 1/2 *(B1[i+1,j,k] + B1[i,j,k]) 	              # Magnetic field component in the x-direction
        B² = 1/2 *(B2[i,j+1,k] + B2[i,j,k]) 	       	      # Magnetic field component in the y-direction
        B³ = 1/2 *(B3[i,j,k+1] + B3[i,j,k]) 		      # Magnetic field component in the z-direction
        
        # Contravariant time component of the four-velocity
        u⁰ = sqrt(u¹*u¹ + u²*u² + u³*u³ + T(1))

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
        enthalpy = ρ + Γ*u + bsq
        
        # Conserved quantities in relativistic magnetohydrodynamics 
        U[1,i,j,k] = ρ*u⁰                                                  # Conserved mass density (D)
        U[2,i,j,k] = enthalpy*u⁰*u₀ + (Γ - T(1))*u + bsq/2 - b⁰*b₀            # Energy density (Q₀)
        U[3,i,j,k] = enthalpy*u⁰*u₁ - b⁰*b₁                                # Conserved momentum density in the x-direction (Q₁)    
        U[4,i,j,k] = enthalpy*u⁰*u₂ - b⁰*b₂                                # Conserved momentum density in the y-direction (Q₂)
        U[5,i,j,k] = enthalpy*u⁰*u₃ - b⁰*b₃                                # Conserved momentum density in the z-direction (Q₃)
    end
end


@inline function PtoU(P::AbstractArray{T}, U::AbstractArray{T}) where T<:Real
    @inbounds begin
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
	    u⁰ = sqrt(u¹*u¹ + u²*u² + u³*u³ + T(1))

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
	    enthalpy = ρ + Γ*u + bsq
	    
	    # Conserved quantities in relativistic magnetohydrodynamics 
	    U[1] = ρ*u⁰                                                  # Conserved mass density (D)
	    U[2] = enthalpy*u⁰*u₀ + (Γ - T(1))*u + bsq/2 - b⁰*b₀            # Energy density (Q₀)
	    U[3] = enthalpy*u⁰*u₁ - b⁰*b₁                                # Conserved momentum density in the x-direction (Q₁)    
	    U[4] = enthalpy*u⁰*u₂ - b⁰*b₂                                # Conserved momentum density in the y-direction (Q₂)
	    U[5] = enthalpy*u⁰*u₃ - b⁰*b₃                                # Conserved momentum density in the z-direction (Q₃)
	end
end

@inline function PtoU_kernel!(P::AbstractArray{T}, B1::AbstractArray{T}, B2::AbstractArray{T}, B3::AbstractArray{T}, U::AbstractArray{T}) where T<:Real
	i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
	k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
	Nx = size(P, 2)
	Ny = size(P, 3)
	Nz = size(P, 4)
	if 4 <= i <= Nx-3 && 4 <= j <= Ny-3 && 4 <= k <= Nz-3
		function_PtoU_kernel!(P, B1, B2, B3, U, i, j, k)
	end
return
end
