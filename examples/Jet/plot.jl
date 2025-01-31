using CairoMakie 
using HDF5
num = 40


fig1 = Figure(size = (1920,1080))
fig2 = Figure(size = (1920,1080))
ax1 = Axis(fig1[1, 1], title = "Density", xlabel = "X", ylabel = "Y")
ax2 = Axis(fig1[1, 2], title = "Vy", xlabel = "X", ylabel = "Y")
ax3 = Axis(fig2[1, 1], title = "Temperature", xlabel = "X", ylabel = "Y")
ax4 = Axis(fig2[1, 2], title = "Gamma", xlabel = "X", ylabel = "Y")

vel_max = 0.3

data = h5open(ARGS[1]*"/dump1.h5","r")
dx = data["grid"][1]
dy = data["grid"][2]
_,X_tot,Y_tot = size(data["data"])
X = (0,X_tot * dx)
Y = (0,Y_tot * dy)
gamma = sqrt.(data["data"][3,:,:] .^ 2 +data["data"][4,:,:] .^2 .+ 1.)
min_val = minimum(log10.(data["data"][1,:,:]))
max_val = maximum(log10.(data["data"][1,:,:]))
println(min_val, " ",max_val)
tmin = maximum(log10.(data["data"][2,div(X_tot,4),:] ./ data["data"][1,div(X_tot,4),:]))
tmax = tmin + 0.1
println(tmin)
hm1 = image!(ax1,X,Y,log10.(data["data"][1,:,:]), colorrange = (min_val, max_val), colormap = :viridis)
hm2 = image!(ax2,X,Y,data["data"][4,:,:] ./ gamma, colorrange = (-vel_max, vel_max), colormap = Reverse(:RdYlBu))
hm3 = image!(ax3,X,Y,log10.(data["data"][2,:,:] ./ data["data"][1,:,:]), colorrange = (tmin, tmax), colormap = :amp)
hm4 = image!(ax4,X,Y,log10.(gamma), colorrange = (0, -log10(sqrt(1-0.3^2))), colormap = :viridis)

close(data)

record(fig1, "Jet_MPI_CUDA_1.mp4", 1:num; framerate = 3) do i
    println("first ",i)
    data = h5open(ARGS[1]*"/dump"*string(i)*".h5","r")
    hm1[3] = log10.(data["data"][1,:,:])
    gamma = sqrt.(data["data"][3,:,:] .^ 2 + data["data"][4,:,:] .^2 .+ 1.)
    hm2[3] = data["data"][4,:,:] ./ gamma
    #hm3[3] = log10.(data["data"][2,:,:] ./ data["data"][1,:,:])
    #hm4[3] = log10.(gamma)
    close(data)
end

record(fig2, "Jet_MPI_CUDA_2.mp4", 1:num; framerate = 3) do i
    println("second ",i)
    data = h5open(ARGS[1]*"/dump"*string(i)*".h5","r")
    #hm1[3] = log10.(data["data"][1,:,:])
    gamma = sqrt.(data["data"][3,:,:] .^ 2 + data["data"][4,:,:] .^2 .+ 1.)
    #hm2[3] = data["data"][4,:,:] ./ gamma
    hm3[3] = log10.(data["data"][2,:,:] ./ data["data"][1,:,:])
    hm4[3] = log10.(gamma)
    close(data)
end
