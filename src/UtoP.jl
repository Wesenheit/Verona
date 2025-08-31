@inline function LU_dec_2D!(flat_matrix::AbstractArray{T}, target::AbstractArray{T}, x::AbstractArray{T}) where T<:Real 
    @inbounds begin
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
end


@inline function function_UtoP_kernel!(U::AbstractArray{T}, P::AbstractArray{T}, B1::AbstractArray{T}, B2::AbstractArray{T}, B3::AbstractArray{T}, i, j, k) where T<:Real 
    @inbounds begin
        
        D  = U[1,i, j, k]
        Q₀ = U[2,i, j, k]
        Q₁ = U[3,i, j, k]
        Q₂ = U[4,i, j, k]
        Q₃ = U[5,i, j, k]
        B¹ = T(0.5) *(B1[i+1,j,k] + B1[i,j,k]) 		# Magnetic field component in the x-direction
        B² = T(0.5) *(B2[i,j+1,k] + B2[i,j,k]) 		# Magnetic field component in the y-direction
        B³ = T(0.5) *(B3[i,j,k+1] + B3[i,j,k]) 		# Magnetic field component in the z-direction
        
        #Useful Values
        S²  = Q₁*Q₁ + Q₂*Q₂ + Q₃*Q₃
        BSQ = B¹*B¹ + B²*B² + B³*B³
        QᵢBⁱ   = Q₁*B¹ + Q₂*B² + Q₃*B³
        α   = (Γ - T(1))/Γ
        
        #Convergence indicators
        convergence_1DW       = false
        convergence_2D        = false
        convergence_Dekker    = false

        #Epsilon value
        ε = T(1e-14)

        #Guess primitive variables
        ρ_guess  = P[1,i,j,k] #Density
        u_guess  = P[2,i,j,k] #Internal Energy 
        u¹_guess = P[3,i,j,k] #Contravariant Four-velocity in x-direction
        u²_guess = P[4,i,j,k] #Contravariant Four-velocity in y-direction
        u³_guess = P[5,i,j,k] #Contravariant Four-velocity in z-direction 
        
        #Guess useful values
        γ_guess  = sqrt(u¹_guess*u¹_guess + u²_guess*u²_guess + u³_guess*u³_guess + T(1))
        w_guess  = ρ_guess + Γ * u_guess
        v²_guess = (u¹_guess*u¹_guess + u²_guess*u²_guess + u³_guess*u³_guess) / γ_guess^2

        # v² less than c² and larger than 0
        if v²_guess >= T(1)                                      
            v²_guess = T(1) - ε
        elseif v²_guess < T(0)
            v²_guess = ε
        end
        
        #1DW METHOD (Noble et al. 2006)
        if !convergence_1DW 
            # Boundary for W
            W = w_guess * γ_guess^2
            W_min = sqrt(S²) * (T(1) + ε) #Approximately
            W_max = T(1e30)     

            # Initial condition, with v²<1
            while (S²*W^2 + (QᵢBⁱ)^2 *(BSQ +2*W))/((BSQ + W)^2 *W^2) >= T(1) && W < W_max
                W *= T(10)
            end

            #Main loop 1DW Newton–Raphson method
            for _ in 1:N_ITER
                
                # W should be greater than W_min
                if W < W_min
                    W = W_min       
                end

                # Function for v²
                v² = (S²*W^2 + (QᵢBⁱ)^2 *(BSQ +T(2)*W))/((BSQ + W)^2 *W^2)

                # v² less than c² and larger than 0             
                if v² < T(0.0)
                    v² = ε
                elseif v² > T(1) -  ε
                    v² = T(1) -  ε
                end	        
                
                # Main function + jacobian
                buff_fun = Q₀ + BSQ/T(2) *(1 + v²) -((QᵢBⁱ)^2 /(2*W^2)) + W - α*(W*(T(1) - v²) - D*sqrt(T(1) - v²))
                d_v²_dW  = T(-2) * ((BSQ)^2 * (QᵢBⁱ)^2 + T(3) * BSQ * (QᵢBⁱ)^2 * W + T(3) * (QᵢBⁱ)^2 * W^2 + S² * W^3)/(W^3 * (BSQ + W)^3)
                buff_jac = BSQ/T(2) *d_v²_dW  + ((QᵢBⁱ)^2 /(W^3)) + T(1) - α * ((T(1) - v²) - W * d_v²_dW + D * d_v²_dW / (T(2) * sqrt(T(1) - v²)))                
                
                #Newton–Raphson method evolution 
                ΔW = buff_fun / buff_jac
                W_proposed = W - ΔW

                if W_proposed < W_min
                    W = T(0.5) * (W + W_min)
                else
                    W = W_proposed
                end
            
                if ΔW < T(0)
                    ΔW = -ΔW
                end  

                #Convergence condition
                if ΔW^2 < TOL^2   
                    convergence_1DW = true
                    break
                end
		
		    end
        end

        #2D METHOD (Noble et al. 2006)
        if !convergence_1DW             
            buff_fun_2D = MVector{2, T}(undef)
            buff_jac_2D = MVector{4, T}(undef)
            buff_out_2D = MVector{2, T}(undef)

            #W and v² guess from previous step
            W  = w_guess * γ_guess^2
            v² = (u¹_guess*u¹_guess + u²_guess*u²_guess + u³_guess*u³_guess) / γ_guess^2

            #Main loop 2D Newton–Raphson method
            for _ in 1:N_ITER 
                W_old  = W
                v²_old = v²

                buff_fun_2D[1] = v²*(BSQ+W)^2 - S² - (QᵢBⁱ)^2 *(BSQ+2*W)/(W^2)
                buff_fun_2D[2] = Q₀ + W + BSQ/T(2) *(T(1)+v²) -(QᵢBⁱ)^2 /(T(2)*W^2) - α*(W*(T(1) - v²) - D*sqrt(max(T(0),T(1) - v²)))

                buff_jac_2D[1] = T(2)*v²*(BSQ+W) + T(2)*(QᵢBⁱ)^2 * (BSQ+W)/W^3 
                buff_jac_2D[2] = T(1) + (QᵢBⁱ)^2 / W^3 - α*(T(1) - v²)
                buff_jac_2D[3] = (BSQ+W)^2 
                buff_jac_2D[4] = BSQ/T(2) + α*W - α*D/(T(2)*sqrt(max(T(0),T(1) - v²)))
                
                LU_dec_2D!(buff_jac_2D, buff_fun_2D, buff_out_2D)
                
                ΔW  = buff_out_2D[1]
                Δv² = buff_out_2D[2]
                
                W  = W  - ΔW
                v² = v² - Δv²        
                
                if ΔW^2 + Δv²^2 < TOL^2
                    convergence_2D = true
                    break
                end                
            end
        end

        #1DW METHOD with Dekker Method (Noble et al. 2006)
        if !convergence_1DW && !convergence_2D 
            
            #Starting Values
            W_min = max(ε, sqrt(S²))
            W_max = W_min * T(10.0)

            v²_min = (S²*W_min^2 + (QᵢBⁱ)^2 *(BSQ +T(2)*W_min))/((BSQ + W_min)^2 *W_min^2)
            v²_max = (S²*W_max^2 + (QᵢBⁱ)^2 *(BSQ +T(2)*W_max))/((BSQ + W_max)^2 *W_max^2)

            fun_min = Q₀ + BSQ/T(2) *(1 + v²_min) -((QᵢBⁱ)^2 /(T(2)*W_min^2)) + W_min - α*(W_min*(1 - v²_min) - D*sqrt(max(T(0),T(1) - v²_min)))
            fun_max = Q₀ + BSQ/T(2) *(1 + v²_max) -((QᵢBⁱ)^2 /(T(2)*W_max^2)) + W_max - α*(W_max*(1 - v²_max) - D*sqrt(max(T(0),T(1) - v²_max)))

            #Looking for the root
            count = 0
            while fun_min*fun_max > 0 && count < 1000
                W_max  *= 10.0
                v²_max = (S²*W_max^2 + (QᵢBⁱ)^2 *(BSQ +T(2)*W_max))/((BSQ + W_max)^2 *W_max^2)
                fun_max = Q₀ + BSQ/T(2) *(T(1) + v²_max) -((QᵢBⁱ)^2 /(T(2)*W_max^2)) + W_max - α*(W_max*(T(1) - v²_max) - D*sqrt(max(T(0),T(1) - v²_max)))
                count += 1
            end
            
            
            if fun_min * fun_max <= 0
                for _ in 1:N_ITER
                    W_sec = W_max - fun_max*(W_max - W_min)/(fun_max - fun_min + ε)

                    if W_sec > W_min && W_sec < W_max
                        W_trial = W_sec
                    else
                        W_trial = T(0.5)*(W_min + W_max)
                    end

                    v²_trial  = (S²*W_trial^2 + (QᵢBⁱ)^2*(BSQ + T(2)*W_trial))/((BSQ + W_trial)^2 * W_trial^2)
                    fun_trial = Q₀ + BSQ/T(2)*(1 + v²_trial) - (QᵢBⁱ)^2/(2*W_trial^2) + W_trial - α*(W_trial*(T(1) - v²_trial) - D*sqrt(max(T(0),T(1) - v²_trial)))

                    if fun_trial^2 < TOL^2
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
                #W = W_max
                #convergence_Dekker = true
            end
        end        



        if convergence_1DW || convergence_2D || convergence_Dekker
            v² = (S²*W^2 + (QᵢBⁱ)^2 *(BSQ +T(2)*W))/((BSQ + W)^2 *W^2)
            γ  = min(T(1)/sqrt(T(1)-v²), T(50))  	         #LORENTZ FACTOR LIMITER
            ρ  = max(FLOORDENISTY, D/γ) 		             #DENSITY FLOOR 
            u  = max(FLOORINTERNALENERGY, (W / γ^2 - D / γ) / Γ)    #INTERNAL ENERGY FLOOR
            u¹ = γ/(W+BSQ) * (Q₁+ QᵢBⁱ*B¹/W)
            u² = γ/(W+BSQ) * (Q₂+ QᵢBⁱ*B²/W)
            u³ = γ/(W+BSQ) * (Q₃+ QᵢBⁱ*B³/W)
            
            ########################################
            #FLOOR for the magnetization for ρ and u
            ########################################
            
            # Contravariant time component of the four-velocity
            u⁰ = sqrt(u¹*u¹ + u²*u² + u³*u³ + T(1))

            # Covariant components of the four-velocity
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
            bsq = b⁰*b₀ + b¹*b₁ + b²*b₂ + b³*b₃		
            

            ρ = max(ρ, bsq/MAXSIGMARHO)
            u = max(u, bsq/MAXSIGMAUU)          

            P[1,i,j,k] = ρ
            P[2,i,j,k] = u
            P[3,i,j,k] = u¹
            P[4,i,j,k] = u²
            P[5,i,j,k] = u³
        end

        if !convergence_1DW && !convergence_2D && !convergence_Dekker
		  P[1,i,j,k] = FLOORDENISTY
		  P[2,i,j,k] = FLOORINTERNALENERGY
		  P[3,i,j,k] = T(0)
		  P[4,i,j,k] = T(0)
		  P[5,i,j,k] = T(0)
	end

    end
end

@inline function UtoP_kernel!(U, P, B1, B2, B3)
	i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
	k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    
    Nx, Ny, Nz = size(P,2), size(P,3), size(P,4)

    if 4 <= i <= Nx-3 && 4 <= j <= Ny-3 && 4 <= k <= Nz-3
		function_UtoP_kernel!(U, P, B1, B2, B3, i, j, k)
	end
	return
end

