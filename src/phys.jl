@inline function cmin_cmax_MHD(PL::AbstractArray{T}, PR::AbstractArray{T}, direction::Int64) where T<:Real
	@inbounds begin
		λ₊_L = λ₋_L = λ₊_R = λ₋_R = T(0)
		for iter in 1:2
			if iter == 1
				ρ  = PL[1] #Density
				u  = PL[2] #Internal Energy 
				u¹ = PL[3] #Contravariant Four-velocity in x-direction
				u² = PL[4] #Contravariant Four-velocity in y-direction
				u³ = PL[5] #Contravariant Four-velocity in z-direction   
				B¹ = PL[6] #Magnetic field in x-direction
				B² = PL[7] #Magnetic field in y-direction
				B³ = PL[8] #Magnetic field in z-direction
			else 
				ρ  = PR[1] #Density
				u  = PR[2] #Internal Energy 
				u¹ = PR[3] #Contravariant Four-velocity in x-direction
				u² = PR[4] #Contravariant Four-velocity in y-direction
				u³ = PR[5] #Contravariant Four-velocity in z-direction   
				B¹ = PR[6] #Magnetic field in x-direction
				B² = PR[7] #Magnetic field in y-direction
				B³ = PR[8] #Magnetic field in z-direction
			end
			#Contravariant Four-velocities
			u⁰ = sqrt(u¹*u¹ + u²*u² + u³*u³ + T(1)) 

			# Covariant Four-velocities
			u₀ = -u⁰ 
			u₁ =  u¹
			u₂ =  u²
			u₃ =  u³	

			# Contravariant Four-magnetic field
			b⁰ = B¹*u₁ + B²*u₂ + B³*u₃
			b¹ = (B¹ + b⁰*u¹)/u⁰
			b² = (B² + b⁰*u²)/u⁰
			b³ = (B³ + b⁰*u³)/u⁰

			# Covariant Four-magnetic field
			b₀ = -b⁰
			b₁ =  b¹
			b₂ =  b²
			b₃ =  b³		

			#Useful Values            
			bsq = b⁰*b₀ + b¹*b₁ + b²*b₂ + b³*b₃

			# Speed of sounds
			cₛ²  = Γ*(u*(Γ-T(1)))/(ρ + Γ*u)
			vₐ²  = bsq/(bsq + ρ + Γ*u)
			cₘₛ² = (cₛ² + vₐ²*(1-cₛ²))/(1+vₐ²)

			#Velocity
			if     direction   == 1
				v = u¹/u⁰
			elseif direction   == 2
				v = u²/u⁰
			elseif   direction == 3
				v = u³/u⁰
			end

			# Lambda characteristic
			if iter == 1  
				λ₊_L = (v + sqrt(cₘₛ²))/(T(1) + v*sqrt(cₘₛ²))
				λ₋_L = (v - sqrt(cₘₛ²))/(T(1) - v*sqrt(cₘₛ²))
			else
				λ₊_R = (v + sqrt(cₘₛ²))/(T(1) + v*sqrt(cₘₛ²))
				λ₋_R = (v - sqrt(cₘₛ²))/(T(1) - v*sqrt(cₘₛ²))	
			end
		end

		C_max_X =  max(λ₊_L, λ₊_R, T(0))
		C_min_X = -min(λ₋_L, λ₋_R, T(0))

		return C_max_X,C_min_X
	end
end
