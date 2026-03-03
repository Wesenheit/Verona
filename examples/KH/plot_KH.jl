using CairoMakie
using HDF5
if length(ARGS) > 1
    num = parse(Int, ARGS[2])
else
    num=100
end
min_val = -1
max_val = 1
fig = Figure(size = (1920, 1080))
ax = Axis(fig[1, 1], title = "Kelvin-Helmholtz instability", xlabel = "X", ylabel = "Y")

data = h5open(ARGS[1]*"/dump0.h5", "r")
hm = heatmap!(
    ax,
    data["data"][2, :, :, 8],
    colorrange = (min_val, max_val),
    colormap = :magma,
)
Colorbar(fig[2, 1], hm, vertical = false)
close(data)

record(fig, "KH_MPI.mp4", 1:num; framerate = 10) do i
    println(i)
    data = h5open(ARGS[1]*"/dump"*string(i)*".h5", "r")
    hm[1] = data["data"][2, :, :, 8]
    close(data)
end
