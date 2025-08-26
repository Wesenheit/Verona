# Verona - hydrodynamical code for relativistic fluids

Verona is a SRHD code written in pure julia accelerated with the CUDA
and parallelized with the nonblocking MPI communication.

It currently supports HLLC riemann solver with following reconstruction methods:

- MINMOD
- PPM
- WENOZ

![Jet3d](examples/Jet3D/render_gam_den.png)
*3D Jet breakout simulation. Red color indicate a highly relativistic outflow $\Gamma > \sqrt{2}$.

## Requirements
To properly launch the code one needs CUDA-aware MPI with HDF5 that should also be compatible 
with the MPI version used to launch the code. 

## Use guide
Library was designed with simplicity in mind and exposes only the most important functionalities.
The examples of the usage can be found in example directory. 
