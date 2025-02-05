import numpy as np
import h5py as h5
from mayavi import mlab
from sys import argv

mlab.options.offscreen = True
"""
render the lorentz factor of the jet + density
"""

name = argv[1]
idx = name.split("/")
file = h5.File(name,"r")

data = file["data"][()].transpose([3,2,1,0])
dx,dy,dz = file["grid"][()]
_,Nx,Ny,Nz = data.shape
#weird indexing
z = np.arange(Nx) * dx
x = np.arange(Ny) * dy - Ny//2*dy
y = np.arange(Nz) * dz - Nz//2*dz

plane = file["T"][()]/2
max_Z = file["T"][()]

X,Y,Z = np.meshgrid(x,y,z,indexing = "ij")
idx = z < max_Z
gamma = np.sqrt(data[2,:,:,:]**2 + data[3,:,:,:]**2 + data[4,:,:,:]**2 + 1)
density = np.log10(data[0,:,:,:]).transpose([1,2,0])
min_den = -3.5
max_den = np.max(density)
to_render = np.log10(gamma.transpose([1,2,0]))
vmin = 0
vmax = 1

figure = mlab.figure(size = (1920,1080))
print(X.shape,Y.shape,Z.shape,to_render.shape)

src = mlab.pipeline.scalar_field(X[:,:,idx],Y[:,:,idx],Z[:,:,idx],to_render[:,:,idx])
den = mlab.pipeline.scalar_field(X[:,:,idx],Y[:,:,idx],Z[:,:,idx],density[:,:,idx])
cmap = "viridis"

cut_plane2 = mlab.pipeline.scalar_cut_plane(den,
                                plane_orientation='z_axes',
                                vmin=min_den,
                                vmax=max_den,
                                colormap = cmap)
cut_plane2.implicit_plane.origin = (0, 0, 1)
cut_plane2.implicit_plane.widget.enabled = False

vol_obj_gam = mlab.pipeline.volume(src,vmin = np.log10(np.sqrt(2)),vmax = vmax)
vol_obj_den = mlab.pipeline.volume(den,vmin = min_den,vmax = max_den)

from tvtk.util.ctf import ColorTransferFunction
ctf = ColorTransferFunction()
ctf.add_rgb_point(vmax, 1, 0, 0)
vol_obj_gam._volume_property.set_color(ctf)
vol_obj_gam._ctf = ctf
vol_obj_gam.update_ctf = True


#vol_obj_gam._volume_property.set_color(mlab.color_maps.get_colormap("hot"))
mlab.axes(vol_obj_gam)
orient = mlab.orientation_axes()
figure.scene.camera.zoom(1.4)

mlab.savefig(filename="render_gam_den.png",figure = figure)
file.close()
