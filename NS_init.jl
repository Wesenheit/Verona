using CUDA
using CUDA
using Statistics
using Random
using Plots
using Printf
using Base.Threads
using StaticArrays
using LinearAlgebra

include("src/main.jl")

#########CONSTANT####################################
const ghost_cells::Int64    = 3
const Γ::T            = 5/3
const N_ITER::Int64   = 100     
const TOL::T          = 1e-8
const floor::T        = 1e-7 
const MAXSIGMARHO::T  = 100
const MAXSIGMAUU::T   = 100
const GAMMAMAX::T     = 50
const Nx::Int64 = 256 + 2*ghost_cells     
const Ny::Int64 = 256 + 2*ghost_cells 
const Nz::Int64 = 256 + 2*ghost_cells 
const FLOORDENISTY::T = 1e-7
const FLOORINTERNALENERGY::T= 1e-7
const CMAX::T = 0.3

#####################INITIALIZING###############
P_CPU, B1_CPU, B2_CPU, B3_CPU, A1_CPU, A2_CPU, A3_CPU = CPU_alloc()
P, U, U_buffer, B1, B2, B3, B1_buffer, B2_buffer, B3_buffer, F1, F2, F3, F1_CT, F2_CT, F3_CT, EMF1, EMF2, EMF3  =GPU_alloc()
#########PHYSICS################################
const RHO_SCALE = 1e-2
const M_UNIT = 1.9892e33*RHO_SCALE
const CL = 2.99792458e10
const R_NEUTRON_STAR = 1200000
const L0 = R_NEUTRON_STAR
const T_UNIT = CL/L0
const RHO_UNIT = M_UNIT/L0^3
const P_UNIT = M_UNIT * CL*CL/ T_UNIT
const B_UNIT = CL * sqrt(4 * pi * RHO_UNIT )
###################BOX#########################
box_X::T = 4*R_NEUTRON_STAR    
box_Y::T = 4*R_NEUTRON_STAR 
box_Z::T = 4*R_NEUTRON_STAR 
dx = box_X/Nx
dy = box_Y/Ny
dz = box_Z/Nz
###############################################
const Ω_c  = 2π/(0.001)               #centralna prędkość kątowa [1/s]
const A    = R_NEUTRON_STAR/2         #parametr określający stopień różnicowości
const ρ_c  = 1e15
const ρ_amb = 1e11
const ρ_atm = 1e9
##############################################
x0, y0, z0 = box_X/2, box_Y/2, box_Z/2
rng = MersenneTwister(1234)
@threads for i in 1:Nx
	 	for j in 1:Ny
			for k in 1:Nz
				x = (i - 0.5)*dx - x0
				y = (j - 0.5)*dy - y0
				z = (k - 0.5)*dz - z0
				r = sqrt(x^2 + y^2 + z^2)+10
				ρ_cyl = sqrt(x^2 + y^2)
				
				if r < R_NEUTRON_STAR
				        ξ = π * r / R_NEUTRON_STAR
        				ρ = ρ_c * (sin(ξ) / ξ) + ρ_amb
    					Ω_loc = Ω_c# / (1 + (ρ_cyl/A)^2)
					vx = -Ω_c * y 
					vy =  Ω_c * x 
					vz = 0
    				else 
    					ρ = ρ_atm	
    					vx = 0
					vy = 0
					vz = 0
				end
				
			P_CPU[1,i,j,k] = ρ/RHO_UNIT *(1 + 0.01*(2*rand(rng) - 1))
			P_CPU[2,i,j,k] = 2.5e-8 * ρ^(4/3) /P_UNIT *(1 + 0.01*(2*rand(rng) - 1))
			P_CPU[3,i,j,k] = vx/CL
			P_CPU[4,i,j,k] = vy/CL
			P_CPU[5,i,j,k] = vz/CL

		end
	end
end

begin 
println("               ")
println("Neutron star parameters:")
local total_mass = 0
for i in 1:Nx
	for j in 1:Ny
		for k in 1:Nz
		    x = (i - 0.5)*dx - x0
		    y = (j - 0.5)*dy - y0
		    z = (k - 0.5)*dz - z0
    		    r = sqrt(x^2 + y^2 + z^2)
    		    if r < R_NEUTRON_STAR
                    	ρ_code = P_CPU[1, i, j, k]
        		total_mass += ρ_code * RHO_UNIT * dx * dy * dz
    		    end
		end
	end
end
println("Neutron star mass: ",total_mass/1.9892e33 , " [Ms]")
println("Neutron star radius: ", R_NEUTRON_STAR /100000 , " [km]")
println("               ")
end

println("P_UNIT: ",P_UNIT)
println("T_UNIT: ",T_UNIT)

#println("rho max:", maximum(P_CPU[1,:,:,:]))
#println("rho min:", minimum(P_CPU[1,:,:,:]))
const m0 = 5e29              
const ε1 = 1e-12               

@threads for i in 1:Nx
         	for j in 1:Ny+1
                	for k in 1:Nz+1
            			x = (i - 0.5) * dx - x0
            			y = (j - 1.0) * dy - y0
            			z = (k - 1.0) * dy - z0
            			r3 = (x^2 + y^2 + z^2 + ε1)^(3/2)
            			A1_CPU[i,j,k] = - m0 * y / r3
        end
    end
end

@threads for i in 1:Nx+1
         	for j in 1:Ny
                	for k in 1:Nz+1
			    x = (i - 1.0) * dx - x0
			    y = (j - 0.5) * dy - y0
			    z = (k - 1.0) * dy - z0
			    r3 = (x^2 + y^2 + z^2 + ε1)^(3/2)
            		    A2_CPU[i,j,k] = m0 * x / r3
        end
    end
end

@threads for i in 1:Nx+1
         	for j in 1:Ny+1
                	for k in 1:Nz
            			A3_CPU[i,j,k] = 0.0 
        end
    end
end

compute_B_from_A!(B1_CPU, B2_CPU, B3_CPU, A1_CPU, A2_CPU, A3_CPU, dx, dy, dz)
println("Rescalling magnetic field...")

begin
local max_sigma = 0
@threads for i in 1:Nx
    for j in 1:Ny
        for k in 1:Nz
            Bx = 0.5 * (B1_CPU[i,j,k] + B1_CPU[i+1,j,k])
            By = 0.5 * (B2_CPU[i,j,k] + B2_CPU[i,j+1,k])
            Bz = 0.5 * (B3_CPU[i,j,k] + B3_CPU[i,j,k+1])
            B_squared = Bx^2 + By^2 + Bz^2
            rho = P_CPU[1,i,j,k]
            sigma = B_squared / rho
            if sigma > max_sigma
                max_sigma = sigma
            end
        end
    end
end
println("Maximum Magnetisation (sigma):", max_sigma)
MAX_MAG = 1
scale_factor = sqrt(MAX_MAG / max_sigma)  

@threads for i in 1:Nx+1
	 	for j in 1:Ny
			for k in 1:Nz
    				B1_CPU[i,j,k] *= scale_factor
		end
	end
end
@threads for i in 1:Nx
	 	for j in 1:Ny+1
			for k in 1:Nz
    				B2_CPU[i,j,k] *= scale_factor
		end
	end
end
@threads for i in 1:Nx
	 	for j in 1:Ny
			for k in 1:Nz+1
    				B3_CPU[i,j,k] *= scale_factor
		end
	end
end

end

#B1_CPU .=0
#B2_CPU .=0
#B3_CPU .=0

println("Copying initial conditions to GPU...")
copyto!(P,  P_CPU)
copyto!(B1, B1_CPU)
copyto!(B2, B2_CPU)
copyto!(B3, B3_CPU)

box_X = box_X/ L0 
box_Y = box_Y/ L0
box_Z = box_Z/ L0

dx = box_X/Nx
dy = box_Y/Ny
dz = box_Z/Nz

println("Initialization complete!")
  
const GPU_THREADS = (6, 6, 6)

BLOCKS_GPU  = (cld(Nx,GPU_THREADS[1]), cld(Ny,GPU_THREADS[2]), cld(Nz,GPU_THREADS[3]))

HLLE()
