@inline function PtoF₁(P::AbstractVector{T}, F₁::AbstractVector{T}) where T<:Real
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

	    # Fluxes in x-direction
	    F₁[1] = ρ*u¹
	    F₁[2] = enthalpy*u¹*u₀ - b¹*b₀
	    F₁[3] = enthalpy*u¹*u₁ + (Γ - T(1))*u + bsq/2 - b¹*b₁
	    F₁[4] = enthalpy*u¹*u₂ - b¹*b₂
	    F₁[5] = enthalpy*u¹*u₃ - b¹*b₃

    end
end


@inline function PtoF₂(P::AbstractVector{T}, F₂::AbstractVector{T}) where T<:Real
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

	    # Fluxes in y-direction
	    F₂[1] = ρ*u²
	    F₂[2] = enthalpy*u²*u₀ - b²*b₀
	    F₂[3] = enthalpy*u²*u₁ - b²*b₁
	    F₂[4] = enthalpy*u²*u₂ + (Γ - T(1))*u + bsq/2 - b²*b₂
	    F₂[5] = enthalpy*u²*u₃ - b²*b₃
    end
end


@inline function PtoF₃(P::AbstractVector{T}, F₃::AbstractVector{T}) where T<:Real
    @inbounds begin
	    #Primitive variables
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

	    # Fluxes in z-direction
	    F₃[1] = ρ*u³
	    F₃[2] = enthalpy*u³*u₀ - b³*b₀
	    F₃[3] = enthalpy*u³*u₁ - b³*b₁
	    F₃[4] = enthalpy*u³*u₂ - b³*b₂
	    F₃[5] = enthalpy*u³*u₃ + (Γ - T(1))*u + bsq/2 - b³*b₃
    end
end


@inline function Flux_X_kernel!(Flux_X::AbstractArray{T}, P::AbstractArray{T}, B1::AbstractArray{T}, B2::AbstractArray{T}, B3::AbstractArray{T}, recon::Int64) where T<:Real
	i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
	k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
		PL = MVector{8, T}(undef)
		PR = MVector{8, T}(undef)
		UL = MVector{5, T}(undef)
		UR = MVector{5, T}(undef)
		FL = MVector{5, T}(undef)
		FR = MVector{5, T}(undef)
		Nx, Ny, Nz = size(P,2), size(P,3), size(P,4)
		if 4 <= i && i <= Nx-2 && 4 <= j && j <= Ny-3 && 4 <= k && k <= Nz-3			
			for idx in 1:5
				q_i   = P[idx,i-1,j,k]
				q_im1 = P[idx,i-2,j,k]
				q_im2 = P[idx,i-3,j,k]
				q_ip1 = P[idx,i,j,k]
				q_ip2 = P[idx,i+1,j,k]

				if recon == 1
					Q_D,Q_U = MINMOD(q_im2,q_im1,q_i,q_ip1,q_ip2)
				elseif recon == 2
					Q_D,Q_U = PPM(q_im2,q_im1,q_i,q_ip1,q_ip2)
				elseif recon == 3
					Q_D,Q_U = WENOZ(q_im2,q_im1,q_i,q_ip1,q_ip2)
				end
					
				PL[idx] = Q_U 
			end

			PL[6] = B1[i,j,k]

			q_i   = T(0.5) *(B2[i-1,j+1,k] + B2[i-1,j,k]) 
			q_im1 = T(0.5) *(B2[i-2,j+1,k] + B2[i-2,j,k]) 
			q_im2 = T(0.5) *(B2[i-3,j+1,k] + B2[i-3,j,k])
			q_ip1 = T(0.5) *(B2[i,  j+1,k] + B2[i,j,  k])   
			q_ip2 = T(0.5) *(B2[i+1,j+1,k] + B2[i+1,j,k]) 
			if     recon == 1
				Q_D,Q_U = MINMOD(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 2
				Q_D,Q_U = PPM(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 3
				Q_D,Q_U = WENOZ(q_im2,q_im1,q_i,q_ip1,q_ip2)
			end
					
			PL[7] = Q_U
				
			q_i   = T(0.5) *(B3[i-1,j,k+1] + B3[i-1,j,k]) 
			q_im1 = T(0.5) *(B3[i-2,j,k+1] + B3[i-2,j,k]) 
			q_im2 = T(0.5) *(B3[i-3,j,k+1] + B3[i-3,j,k]) 
			q_ip1 = T(0.5) *(B3[i,  j,k+1] + B3[i  ,j,k])   
			q_ip2 = T(0.5) *(B3[i+1,j,k+1] + B3[i+1,j,k]) 

			if     recon == 1
				Q_D,Q_U = MINMOD(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 2
				Q_D,Q_U = PPM(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 3
				Q_D,Q_U = WENOZ(q_im2,q_im1,q_i,q_ip1,q_ip2)
			end

			PL[8] = Q_U

			for idx in 1:5
				q_i   = P[idx,i,j,  k]
				q_im1 = P[idx,i-1,j,k]
				q_im2 = P[idx,i-2,j,k]
				q_ip1 = P[idx,i+1,j,k]
				q_ip2 = P[idx,i+2,j,k]
				if     recon == 1
					Q_D,Q_U = MINMOD(q_im2,q_im1,q_i,q_ip1,q_ip2)
				elseif recon == 2
					Q_D,Q_U = PPM(q_im2,q_im1,q_i,q_ip1,q_ip2)
				elseif recon == 3
					Q_D,Q_U = WENOZ(q_im2,q_im1,q_i,q_ip1,q_ip2)
				end
				PR[idx] = Q_D 
			end

			PR[6] = B1[i,j,k]
			
			q_i =   T(0.5) *(B2[i,j,k]   + B2[i,j+1,k])
			q_im1 = T(0.5) *(B2[i-1,j,k] + B2[i-1,j+1,k]) 
			q_im2 = T(0.5) *(B2[i-2,j,k] + B2[i-2,j+1,k]) 
			q_ip1 = T(0.5) *(B2[i+1,j,k] + B2[i+1,j+1,k]) 
			q_ip2 = T(0.5) *(B2[i+2,j,k] + B2[i+2,j+1,k]) 
				
			if recon == 1
				Q_D,Q_U = MINMOD(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 2
				Q_D,Q_U = PPM(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 3
				Q_D,Q_U = WENOZ(q_im2,q_im1,q_i,q_ip1,q_ip2)
			end

			PR[7] = Q_D
				
				
			q_i =   T(0.5) *(B3[i,j,k] + B3[i,j,k+1]) 
			q_im1 = T(0.5) *(B3[i-1,j,k] + B3[i-1,j,k+1]) 
			q_im2 = T(0.5) *(B3[i-2,j,k] + B3[i-2,j,k+1])
			q_ip1 = T(0.5) *(B3[i+1,j,k] + B3[i+1,j,k+1]) 
			q_ip2 = T(0.5) *(B3[i+2,j,k] + B3[i+2,j,k+1]) 
			
			if recon == 1
				Q_D,Q_U = MINMOD(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 2
				Q_D,Q_U = PPM(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 3
				Q_D,Q_U = WENOZ(q_im2,q_im1,q_i,q_ip1,q_ip2)
			end
				
			PR[8] = Q_D
		
			for idx in 1:2
				PL[idx] = max(floor,PL[idx])
				PR[idx] = max(floor,PR[idx])
			end
			
			PtoU(PL,UL)
			PtoU(PR,UR)
        	
			PtoF₁(PR,FR)
			PtoF₁(PL,FL)

			C_max_X, C_min_X = cmin_cmax_MHD(PL, PR, 1)

			ε = T(1e-10)
			denom = C_max_X + C_min_X
			if !isfinite(denom) || abs(denom) < ε
			    splus = copysign(ε, denom)
			else
			    splus = denom
			end
		       
        		if C_max_X < 0 
				for idx in 1:5
					Flux_X[idx,i,j,k] =  FR[idx]
				end
			elseif C_min_X < 0 
				for idx in 1:5
					Flux_X[idx,i,j,k] =  FL[idx] 
				end
			else
				for idx in 1:5
					Flux_X[idx,i,j,k] = (FR[idx] * C_min_X + FL[idx] * C_max_X - C_max_X * C_min_X * (UR[idx] - UL[idx])) / (splus)
				end
			end			
		end
return
end



@inline function Flux_Y_kernel!(Flux_Y::AbstractArray{T}, P::AbstractArray{T}, B1::AbstractArray{T}, B2::AbstractArray{T}, B3::AbstractArray{T}, recon::Int64) where T<:Real
	i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
	k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
		PL = MVector{8, T}(undef)
		PR = MVector{8, T}(undef)
		UL = MVector{5, T}(undef)
		UR = MVector{5, T}(undef)
		FL = MVector{5, T}(undef)
		FR = MVector{5, T}(undef)
		Nx, Ny, Nz = size(P,2), size(P,3), size(P,4)
		if 4 <= i && i <= Nx-3 && 4 <= j && j <= Ny-2 && 4 <= k && k <= Nz-3			
			for idx in 1:5
				q_i   = P[idx,i,j-1,k]
				q_im1 = P[idx,i,j-2,k]
				q_im2 = P[idx,i,j-3,k]
				q_ip1 = P[idx,i,j,k]
				q_ip2 = P[idx,i,j+1,k]

				if recon == 1
					Q_D,Q_U = MINMOD(q_im2,q_im1,q_i,q_ip1,q_ip2)
				elseif recon == 2
					Q_D,Q_U = PPM(q_im2,q_im1,q_i,q_ip1,q_ip2)
				elseif recon == 3
					Q_D,Q_U = WENOZ(q_im2,q_im1,q_i,q_ip1,q_ip2)
				end
					
				PL[idx] = Q_U 
			end

			q_i   = T(0.5) *(B1[i+1,j-1,k] + B1[i,j-1,k]) 
			q_im1 = T(0.5) *(B1[i+1,j-2,k] + B1[i,j-2,k]) 
			q_im2 = T(0.5) *(B1[i+1,j-3,k] + B1[i,j-3,k])
			q_ip1 = T(0.5) *(B1[i+1,j,k]   + B1[i,j,k])   
			q_ip2 = T(0.5) *(B1[i+1,j+1,k] + B1[i,j+1,k]) 

			if     recon == 1
				Q_D,Q_U = MINMOD(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 2
				Q_D,Q_U = PPM(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 3
				Q_D,Q_U = WENOZ(q_im2,q_im1,q_i,q_ip1,q_ip2)
			end
					
			PL[6] = Q_U

			PL[7] = B2[i,j,k]
				
			q_i   = T(0.5) *(B3[i,j-1,k+1] + B3[i,j-1,k]) 
			q_im1 = T(0.5) *(B3[i,j-2,k+1] + B3[i,j-2,k]) 
			q_im2 = T(0.5) *(B3[i,j-3,k+1] + B3[i,j-3,k]) 
			q_ip1 = T(0.5) *(B3[i,  j,k+1] + B3[i,j,k])   
			q_ip2 = T(0.5) *(B3[i,j+1,k+1] + B3[i,j+1,k]) 

			if     recon == 1
				Q_D,Q_U = MINMOD(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 2
				Q_D,Q_U = PPM(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 3
				Q_D,Q_U = WENOZ(q_im2,q_im1,q_i,q_ip1,q_ip2)
			end

			PL[8] = Q_U

			for idx in 1:5
				q_i   = P[idx,i,j,  k]
				q_im1 = P[idx,i,j-1,k]
				q_im2 = P[idx,i,j-2,k]
				q_ip1 = P[idx,i,j+1,k]
				q_ip2 = P[idx,i,j+2,k]
				if     recon == 1
					Q_D,Q_U = MINMOD(q_im2,q_im1,q_i,q_ip1,q_ip2)
				elseif recon == 2
					Q_D,Q_U = PPM(q_im2,q_im1,q_i,q_ip1,q_ip2)
				elseif recon == 3
					Q_D,Q_U = WENOZ(q_im2,q_im1,q_i,q_ip1,q_ip2)
				end
				PR[idx] = Q_D 
			end

			
			q_i =   T(0.5) *(B1[i+1,j,k]   + B1[i,j,k])
			q_im1 = T(0.5) *(B1[i+1,j-1,k] + B1[i,j-1,k]) 
			q_im2 = T(0.5) *(B1[i+1,j-2,k] + B1[i,j-2,k]) 
			q_ip1 = T(0.5) *(B1[i+1,j+1,k] + B1[i,j+1,k]) 
			q_ip2 = T(0.5) *(B1[i+1,j+2,k] + B1[i,j+2,k]) 
				
			if recon == 1
				Q_D,Q_U = MINMOD(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 2
				Q_D,Q_U = PPM(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 3
				Q_D,Q_U = WENOZ(q_im2,q_im1,q_i,q_ip1,q_ip2)
			end

			PR[6] = Q_D
				
			PR[7] = B2[i,j,k]


			q_i =   T(0.5) *(B3[i,j,k]   + B3[i,j,k+1]) 
			q_im1 = T(0.5) *(B3[i,j-1,k] + B3[i,j-1,k+1]) 
			q_im2 = T(0.5) *(B3[i,j-2,k] + B3[i,j-2,k+1])
			q_ip1 = T(0.5) *(B3[i,j+1,k] + B3[i,j+1,k+1]) 
			q_ip2 = T(0.5) *(B3[i,j+2,k] + B3[i,j+2,k+1]) 
			
			if recon == 1
				Q_D,Q_U = MINMOD(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 2
				Q_D,Q_U = PPM(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 3
				Q_D,Q_U = WENOZ(q_im2,q_im1,q_i,q_ip1,q_ip2)
			end
				
			PR[8] = Q_D
		
			for idx in 1:2
				PL[idx] = max(floor,PL[idx])
				PR[idx] = max(floor,PR[idx])
			end
			
			PtoU(PL,UL)
			PtoU(PR,UR)
        	
			PtoF₂(PR,FR)
			PtoF₂(PL,FL)

			C_max_X, C_min_X = cmin_cmax_MHD(PL, PR, 2)
		       	
			ε = T(1e-10)
			denom = C_max_X + C_min_X
			if !isfinite(denom) || abs(denom) < ε
			    splus = copysign(ε, denom)
			else
			    splus = denom
			end
			
        	if C_max_X < 0 
				for idx in 1:5
					Flux_Y[idx,i,j,k] =  FR[idx]
				end
			elseif C_min_X < 0 
				for idx in 1:5
					Flux_Y[idx,i,j,k] =  FL[idx] 
				end
			else
				for idx in 1:5
					Flux_Y[idx,i,j,k] = (FR[idx] * C_min_X + FL[idx] * C_max_X - C_max_X * C_min_X * (UR[idx] - UL[idx])) / (splus)
				end
			end			
		end
return
end

@inline function Flux_Z_kernel!(Flux_Z::AbstractArray{T}, P::AbstractArray{T}, B1::AbstractArray{T}, B2::AbstractArray{T}, B3::AbstractArray{T}, recon::Int64) where T<:Real
	i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
	k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
		PL = MVector{8, T}(undef)
		PR = MVector{8, T}(undef)
		UL = MVector{5, T}(undef)
		UR = MVector{5, T}(undef)
		FL = MVector{5, T}(undef)
		FR = MVector{5, T}(undef)
		Nx, Ny, Nz = size(P,2), size(P,3), size(P,4)
		if 4 <= i && i <= Nx-3 && 4 <= j && j <= Ny-3 && 4 <= k && k <= Nz-2			
			for idx in 1:5
				q_i   = P[idx,i,j,k-1]
				q_im1 = P[idx,i,j,k-2]
				q_im2 = P[idx,i,j,k-3]
				q_ip1 = P[idx,i,j,k]
				q_ip2 = P[idx,i,j,k+1]

				if recon == 1
					Q_D,Q_U = MINMOD(q_im2,q_im1,q_i,q_ip1,q_ip2)
				elseif recon == 2
					Q_D,Q_U = PPM(q_im2,q_im1,q_i,q_ip1,q_ip2)
				elseif recon == 3
					Q_D,Q_U = WENOZ(q_im2,q_im1,q_i,q_ip1,q_ip2)
				end
					
				PL[idx] = Q_U 
			end

			q_i   = T(0.5) *(B1[i+1,j,k-1] + B1[i,j,k-1]) 
			q_im1 = T(0.5) *(B1[i+1,j,k-2] + B1[i,j,k-2]) 
			q_im2 = T(0.5) *(B1[i+1,j,k-3] + B1[i,j,k-3])
			q_ip1 = T(0.5) *(B1[i+1,j,k]   + B1[i,j,k])   
			q_ip2 = T(0.5) *(B1[i+1,j,k+1] + B1[i,j,k+1]) 

			if     recon == 1
				Q_D,Q_U = MINMOD(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 2
				Q_D,Q_U = PPM(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 3
				Q_D,Q_U = WENOZ(q_im2,q_im1,q_i,q_ip1,q_ip2)
			end
					
			PL[6] = Q_U
				
			q_i   = T(0.5) *(B2[i,j+1,k-1] + B2[i,j,k-1]) 
			q_im1 = T(0.5) *(B2[i,j+1,k-2] + B2[i,j,k-2]) 
			q_im2 = T(0.5) *(B2[i,j+1,k-3] + B2[i,j,k-3]) 
			q_ip1 = T(0.5) *(B2[i,j+1,k]   + B2[i,j,k])   
			q_ip2 = T(0.5) *(B2[i,j+1,k+1] + B2[i,j,k+1]) 

			if     recon == 1
				Q_D,Q_U = MINMOD(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 2
				Q_D,Q_U = PPM(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 3
				Q_D,Q_U = WENOZ(q_im2,q_im1,q_i,q_ip1,q_ip2)
			end
			PL[7] = Q_U
			PL[8] = B3[i,j,k]

			for idx in 1:5
				q_i   = P[idx,i,j,  k]
				q_im1 = P[idx,i,j,k-1]
				q_im2 = P[idx,i,j,k-2]
				q_ip1 = P[idx,i,j,k+1]
				q_ip2 = P[idx,i,j,k+2]
				if     recon == 1
					Q_D,Q_U = MINMOD(q_im2,q_im1,q_i,q_ip1,q_ip2)
				elseif recon == 2
					Q_D,Q_U = PPM(q_im2,q_im1,q_i,q_ip1,q_ip2)
				elseif recon == 3
					Q_D,Q_U = WENOZ(q_im2,q_im1,q_i,q_ip1,q_ip2)
				end
				PR[idx] = Q_D 
			end

			
			q_i =   T(0.5) *(B1[i+1,j,k]   + B1[i,j,k])
			q_im1 = T(0.5) *(B1[i+1,j,k-1] + B1[i,j,k-1]) 
			q_im2 = T(0.5) *(B1[i+1,j,k-2] + B1[i,j,k-2]) 
			q_ip1 = T(0.5) *(B1[i+1,j,k+1] + B1[i,j,k+1]) 
			q_ip2 = T(0.5) *(B1[i+1,j,k+2] + B1[i,j,k+2]) 
				
			if recon == 1
				Q_D,Q_U = MINMOD(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 2
				Q_D,Q_U = PPM(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 3
				Q_D,Q_U = WENOZ(q_im2,q_im1,q_i,q_ip1,q_ip2)
			end

			PR[6] = Q_D

			q_i =   T(0.5) *(B2[i,j+1,k]   + B2[i,j,k]) 
			q_im1 = T(0.5) *(B2[i,j+1,k-1] + B2[i,j,k-1]) 
			q_im2 = T(0.5) *(B2[i,j+1,k-2] + B2[i,j,k-2])
			q_ip1 = T(0.5) *(B2[i,j+1,k+1] + B2[i,j,k+1]) 
			q_ip2 = T(0.5) *(B2[i,j+1,k+2] + B2[i,j,k+2]) 
			
			if recon == 1
				Q_D,Q_U = MINMOD(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 2
				Q_D,Q_U = PPM(q_im2,q_im1,q_i,q_ip1,q_ip2)
			elseif recon == 3
				Q_D,Q_U = WENOZ(q_im2,q_im1,q_i,q_ip1,q_ip2)
			end
				
			PR[7] = Q_D
			
			PR[8] = B3[i,j,k]

			for idx in 1:2
				PL[idx] = max(floor,PL[idx])
				PR[idx] = max(floor,PR[idx])
			end
			
			PtoU(PL,UL)
			PtoU(PR,UR)
        	
			PtoF₃(PR,FR)
			PtoF₃(PL,FL)

			C_max_X, C_min_X = cmin_cmax_MHD(PL, PR, 3)
		       	
			ε = T(1e-10)
			denom = C_max_X + C_min_X
			if !isfinite(denom) || abs(denom) < ε
			    splus = copysign(ε, denom)
			else
			    splus = denom
			end
		       	
        	if C_max_X < 0 
				for idx in 1:5
					Flux_Z[idx,i,j,k] =  FR[idx]
				end
			elseif C_min_X < 0 
				for idx in 1:5
					Flux_Z[idx,i,j,k] =  FL[idx] 
				end
			else
				for idx in 1:5
					Flux_Z[idx,i,j,k] = (FR[idx] * C_min_X + FL[idx] * C_max_X - C_max_X * C_min_X * (UR[idx] - UL[idx])) / (splus)
				end
			end			
		end
return
end
