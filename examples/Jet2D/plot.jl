using CairoMakie 
using HDF5
num = 99


fig1 = Figure(size = (1920,1080))
fig2 = Figure(size = (1920,1080))
ax1 = Axis(fig1[1, 1], title = "Density", xlabel = "Y", ylabel = "X")
ax2 = Axis(fig2[1, 1], title = "Gamma", xlabel = "Y", ylabel = "X")


data = h5open(ARGS[1]*"/dump0.h5","r")
dx = data["grid"][1]
dy = data["grid"][2]
_,X_tot,Y_tot = size(data["data"])
X = (-div(X_tot,2)*dx,div(X_tot,2) * dx)
Y = (0,Y_tot * dy)
gamma = sqrt.(data["data"][3,:,:] .^ 2 +data["data"][4,:,:] .^2 .+ 1.)
gam_max = log10(maximum(gamma))
min_val = minimum(log10.(data["data"][1,:,:]))
max_val = maximum(log10.(data["data"][1,:,:]))
hm1 = image!(ax1,X,Y,log10.(data["data"][1,:,:]), colorrange = (min_val, max_val), colormap = :cork)
hm2 = image!(ax2,X,Y,log10.(gamma), colorrange = (0, gam_max), colormap = :hot)

ax1.title = "Density: "*string(data["T"][])  # Update title
ax2.title = "Gamma: "*string(data["T"][])  # Update title
close(data)

record(fig1, "Jet_MPI_CUDA_1.mp4", 1:num; framerate = 3) do i
    println("first ",i)
    data = h5open(ARGS[1]*"/dump"*string(i)*".h5","r")
    hm1[3] = log10.(data["data"][1,:,:])
    ax1.title = "Density, T="*string(round(data["T"][],sigdigits = 4))  # Update title
    close(data)
end

record(fig2, "Jet_MPI_CUDA_2.mp4", 1:num; framerate = 3) do i
    println("second ",i)
    data = h5open(ARGS[1]*"/dump"*string(i)*".h5","r")
    gamma = sqrt.(data["data"][3,:,:] .^ 2 + data["data"][4,:,:] .^2 .+ 1.)
    hm2[3] = log10.(gamma)
    ax2.title = "Gamma, T="*string(round(data["T"][],sigdigits = 4))  # Update title
    close(data)
end
