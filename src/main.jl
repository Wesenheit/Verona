println("Starting Verona: initializing MHD simulation on GPU...")
using CUDA
using CUDA
using Statistics
using Random
using Plots
using Printf
using Base.Threads
using StaticArrays
using LinearAlgebra

T = Float64

# include main source files
include("phys.jl")
include("nabla.jl")
include("boundary.jl")
include("fluxes.jl")
include("HDF5.jl")
include("Update.jl")
include("UtoP.jl")
include("PtoU.jl")

# include CT-related files
include("CT/EMF.jl")
include("CT/fluxesCT.jl")
include("CT/UpdateB.jl")

# include reconstruction methods
include("Reconstruction/MINMOD.jl")
include("Reconstruction/PPM.jl")
include("Reconstruction/WENOZ.jl")
include("Reconstruction/MC.jl")


function CPU_alloc()
    println("Allocating CPU arrays for primitives and fields...")
    P_CPU  = zeros(T, 5, Nx, Ny, Nz)
    B1_CPU = zeros(T, Nx+1, Ny, Nz)
    B2_CPU = zeros(T, Nx, Ny+1, Nz)
    B3_CPU = zeros(T, Nx, Ny, Nz+1)
    A1_CPU = zeros(T, Nx, Ny+1, Nz+1)
    A2_CPU = zeros(T, Nx+1, Ny, Nz+1)
    A3_CPU = zeros(T, Nx+1, Ny+1, Nz)
    return P_CPU, B1_CPU, B2_CPU, B3_CPU, A1_CPU, A2_CPU, A3_CPU
end

function GPU_alloc()
	println("Allocating GPU arrays and buffers...")
	P         = CUDA.zeros(T, 5, Nx, Ny,  Nz)
	U         = CUDA.zeros(T, 5, Nx, Ny,  Nz)
	U_buffer  = CUDA.zeros(T, 5, Nx, Ny,  Nz)
	B1        = CUDA.zeros(T, Nx+1, Ny,   Nz)
	B2        = CUDA.zeros(T, Nx,   Ny+1, Nz)
	B3        = CUDA.zeros(T, Nx,   Ny, Nz+1)
	B1_buffer = CUDA.zeros(T, Nx+1, Ny,   Nz)
	B2_buffer = CUDA.zeros(T, Nx,   Ny+1, Nz)
	B3_buffer = CUDA.zeros(T, Nx,   Ny, Nz+1)
	F1        = CUDA.zeros(T, 5, Nx+1, Ny, Nz)
	F2        = CUDA.zeros(T, 5, Nx, Ny+1, Nz)
	F3        = CUDA.zeros(T, 5, Nx, Ny, Nz+1)
	F1_CT     = CUDA.zeros(T, 3, Nx+1, Ny, Nz)
	F2_CT     = CUDA.zeros(T, 3, Nx, Ny+1, Nz)
	F3_CT     = CUDA.zeros(T, 3, Nx, Ny, Nz+1)
	EMF1      = CUDA.zeros(T, Nx, Ny+1,  Nz+1)
	EMF2      = CUDA.zeros(T, Nx+1, Ny,  Nz+1)
	EMF3      = CUDA.zeros(T, Nx+1, Ny+1,  Nz)
	return P, U, U_buffer, B1, B2, B3, B1_buffer, B2_buffer, B3_buffer, F1, F2, F3, F1_CT, F2_CT, F3_CT, EMF1, EMF2, EMF3 
end

function HLLE()
	dt::T = CMAX / (1/dx + 1/dy + 1/dz)
	tf = 5000
	println("                       ")
	println("Starting calcualtion...")
	println("dt = ",dt)
	println("tf = ",tf)
	zones = Nx * Ny * Nz
	measure_interval = 10
	t_last = time()          
	@cuda threads=GPU_THREADS blocks=BLOCKS_GPU PtoU_kernel!(P, B1, B2, B3, U)
	t = 0.0
	i = 0
	println("                       ")
	while t<tf
		println("Iteration: ",i)
		println("Time elapsed: ",t)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU Flux_X_kernel!(F1, P, B1, B2, B3, 1)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU Flux_Y_kernel!(F2, P, B1, B2, B3, 1)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU Flux_Z_kernel!(F3, P, B1, B2, B3, 1)
		
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU FluxCT_X_kernel!(F1_CT, P, B1, B2, B3, 1)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU FluxCT_Y_kernel!(F2_CT, P, B1, B2, B3, 1)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU FluxCT_Z_kernel!(F3_CT, P, B1, B2, B3, 1)

		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU EMF_X_kernel!(EMF1, F1_CT, F2_CT, F3_CT)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU EMF_Y_kernel!(EMF2, F1_CT, F2_CT, F3_CT)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU EMF_Z_kernel!(EMF3, F1_CT, F2_CT, F3_CT)
		
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU kernel_Update!(U, U_buffer, dt/2, dx, dy, dz, F1, F2, F3)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU kernel_UpdateB!(B1, B2, B3, B1_buffer, B2_buffer, B3_buffer, EMF1, EMF2, EMF3, dt/2, dx, dy, dz)
		
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU open_boundaries_all_B1!(B1_buffer,B2_buffer,B3_buffer)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU open_boundaries_all_B2!(B1_buffer,B2_buffer,B3_buffer)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU open_boundaries_all_B3!(B1_buffer,B2_buffer,B3_buffer)
		
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU UtoP_kernel!(U_buffer, P, B1_buffer, B2_buffer, B3_buffer)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU open_boundaries_all!(P)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU open_boundaries_all_B1!(B1,B2,B3)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU open_boundaries_all_B2!(B1,B2,B3)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU open_boundaries_all_B3!(B1,B2,B3)		
		
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU open_boundaries_all_B1!(B1_buffer,B2_buffer,B3_buffer)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU open_boundaries_all_B2!(B1_buffer,B2_buffer,B3_buffer)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU open_boundaries_all_B3!(B1_buffer,B2_buffer,B3_buffer)

		
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU Flux_X_kernel!(F1, P, B1_buffer, B2_buffer, B3_buffer, 1)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU Flux_Y_kernel!(F2, P, B1_buffer, B2_buffer, B3_buffer, 1)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU Flux_Z_kernel!(F3, P, B1_buffer, B2_buffer, B3_buffer, 1)

		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU FluxCT_X_kernel!(F1_CT, P, B1_buffer, B2_buffer, B3_buffer, 1)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU FluxCT_Y_kernel!(F2_CT, P, B1_buffer, B2_buffer, B3_buffer, 1)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU FluxCT_Z_kernel!(F3_CT, P, B1_buffer, B2_buffer, B3_buffer, 1)
		

		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU EMF_X_kernel!(EMF1, F1_CT, F2_CT, F3_CT)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU EMF_Y_kernel!(EMF2, F1_CT, F2_CT, F3_CT)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU EMF_Z_kernel!(EMF3, F1_CT, F2_CT, F3_CT)

		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU kernel_Update!(U, U, dt, dx, dy, dz, F1, F2, F3)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU kernel_UpdateB!(B1, B2, B3, B1, B2, B3,  EMF1, EMF2, EMF3, dt, dx,dy,dz)
	
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU UtoP_kernel!(U, P, B1, B2, B3)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU open_boundaries_all!(P)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU open_boundaries_all_B1!(B1,B2,B3)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU open_boundaries_all_B2!(B1,B2,B3)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU open_boundaries_all_B3!(B1,B2,B3)		
		
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU open_boundaries_all_B1!(B1_buffer,B2_buffer,B3_buffer)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU open_boundaries_all_B2!(B1_buffer,B2_buffer,B3_buffer)
		@cuda threads=GPU_THREADS blocks=BLOCKS_GPU open_boundaries_all_B3!(B1_buffer,B2_buffer,B3_buffer)

		if i % measure_interval == 0
		    t_now = time()
		    dt_measure = t_now - t_last
		    zps = measure_interval * zones / dt_measure
		    println("zones/s =", zps)
		    t_last = t_now
		end

		if i % 10 ==0
			B1_CPU = Array(B1)
			B2_CPU = Array(B2)
			B3_CPU = Array(B3)
			println( "Total ∇*B = ", divB_inner(B1_CPU, B2_CPU, B3_CPU, dx, dy, dz))
		end
		
		if i % 10 == 0
			P_host = Array(P)
			j0 = fld(Ny, 2)  

			dens = @view P_host[1, :, j0, :]  # (x, z)

			xlims = (0, Nx)
			ylims = (0, Nz)

			default(size = (1920, 1080))

			p = heatmap(
			    dens',
			    aspect_ratio = :equal,
			    xlabel = "i",
			    ylabel = "k",
			    title  = "Density at iter = $i",
			    framestyle = :box,
			    colorbar = true,
			    #clim = clim_fixed,
			    xlims = xlims,
			    ylims = ylims
			)

			fname = @sprintf("frames/density_xz_%05d.png", i)
			savefig(p, fname)
		end
		
		
		
		t+=dt
		i+=1

	end
end
