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
const box_X::T = 1.0    
const box_Y::T = 1.0
const box_Z::T = 1.0

dx = box_X/Nx
dy = box_Y/Ny
dz = box_Z/Nz

#####################INITIALIZING###############
P_CPU, B1_CPU, B2_CPU, B3_CPU, A1_CPU, A2_CPU, A3_CPU = CPU_alloc()
P, U, U_buffer, B1, B2, B3, B1_buffer, B2_buffer, B3_buffer, F1, F2, F3, F1_CT, F2_CT, F3_CT, EMF1, EMF2, EMF3  =GPU_alloc()
#########PHYSICS################################

x0, y0, z0 = box_X/2, box_Y/2, box_Z/2

const γ    = 5/3
const B0   = 1/sqrt(4π)        
const ρ0   = 25/(36π)         
const P0   = 5/(12π)           

@threads for i in 1:Nx
	for j in 1:Ny
		for k in 1:Nz
    x = (i-0.5)*dx
    y = (j-0.5)*dy
    ρ    = ρ0
    E    = P0/(γ - 1)
    vx   = -sin(2π * y)
    vy   =  sin(2π * x)
    vz   =  0.0

    P_CPU[1,i,j,k] = ρ
    P_CPU[2,i,j,k] = E
    P_CPU[3,i,j,k] = vx
    P_CPU[4,i,j,k] = vy
    P_CPU[5,i,j,k] = vz
end
end
end

# ------------------------------------------------------------
const B0 = 1/sqrt(4π)

@threads for i in 1:Nx
	for j in 1:Ny+1
		for k in 1:Nz+1
    A1_CPU[i,j,k] = 0.0
end
end
end

# Ay = 0
@threads for i in 1:Nx+1
	for j in 1:Ny 
	for k in 1:Nz+1
    A2_CPU[i,j,k] = 0.0
end
end
end

@threads for i in 1:Nx+1
	for j in 1:Ny+1
		for k in 1:Nz
    x = (i-1)*dx
    y = (j-1)*dy
    A3_CPU[i,j,k] = B0*( cos(4π*x)/(4π) + cos(2π*y)/(2π) )
end
end
end
compute_B_from_A!(B1_CPU, B2_CPU, B3_CPU, A1_CPU, A2_CPU, A3_CPU, dx, dy, dz)


println("Copying initial conditions to GPU...")
copyto!(P,  P_CPU)

copyto!(B1, B1_CPU)
copyto!(B2, B2_CPU)
copyto!(B3, B3_CPU)

println("Initialization complete!")
  
const GPU_THREADS = (6, 6, 6)

BLOCKS_GPU  = (cld(Nx,GPU_THREADS[1]), cld(Ny,GPU_THREADS[2]), cld(Nz,GPU_THREADS[3]))

HLLE()
