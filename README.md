# Verona - hydrodynamical code for relativistic fluids

Verona is a SRHD code written in pure julia accelerated with the CUDA
and parallelized with the nonblocking MPI communication.

It currently supports HLLC riemann solver with following reconstruction methods:

- MINMOD
- PPM
- WENOZ

![Jet3d](examples/Jet3D/render_gam_den.png)
*Figure 1: 3D Jet breakout simulation.
