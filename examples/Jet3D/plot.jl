using CairoMakie
using HDF5
num = 99


fig1 = Figure(size = (1920, 1080))
fig2 = Figure(size = (1920, 1080))
ax1 = Axis(fig1[1, 1], title = "Density", xlabel = "Y", ylabel = "X")
ax2 = Axis(fig2[1, 1], title = "Gamma", xlabel = "Y", ylabel = "X")


data = h5open(ARGS[1]*"/dump0.h5", "r")
dx = data["grid"][1]
dy = data["grid"][2]
_, X_tot, Y_tot = size(data["data"])
Y = (-div(Y_tot, 2)*dy, div(Y_tot, 2) * dy)
X = (0, X_tot * dx)
Z_slice = div(size(data["data"])[4], 2)

v2 = data["data"][2, :, :, Z_slice].^2 .+
     data["data"][3, :, :, Z_slice].^2 .+
     data["data"][4, :, :, Z_slice].^2

gamma = 1.0 ./ sqrt.(1 .- v2)

gam_max = log10(maximum(gamma))
min_val = minimum(log10.(data["data"][1, :, :, Z_slice]))
max_val = maximum(log10.(data["data"][1, :, :, Z_slice]))
hm1 = image!(
    ax1,
    X,
    Y,
    log10.(data["data"][1, :, :, Z_slice]),
    colorrange = (min_val, max_val),
    colormap = :cork,
)
hm2 = image!(ax2, X, Y, log10.(gamma), colorrange = (0, gam_max), colormap = :hot)

ax1.title = "Density: "*string(data["T"][])  # Update title
ax2.title = "Gamma: "*string(data["T"][])  # Update title
close(data)

record(fig1, "Jet_MPI_CUDA_1.mp4", 1:num; framerate = 3) do i
    println("first ", i)
    data = h5open(ARGS[1]*"/dump"*string(i)*".h5", "r")
    hm1[3] = log10.(data["data"][1, :, :, Z_slice])
    ax1.title = "Density, T="*string(round(data["T"][], sigdigits = 4))  # Update title
    close(data)
end

record(fig2, "Jet_MPI_CUDA_2.mp4", 1:num; framerate = 3) do i
    println("second ", i)
    data = h5open(ARGS[1]*"/dump"*string(i)*".h5", "r")
    gamma = sqrt.(
        data["data"][3, :, :, Z_slice] .^ 2 +
        data["data"][4, :, :, Z_slice] .^ 2 +
        data["data"][5, :, :, Z_slice] .^ 2 .+ 1.0,
    )
    hm2[3] = log10.(gamma)
    ax2.title = "Gamma, T="*string(round(data["T"][], sigdigits = 4))  # Update title
    close(data)
end
