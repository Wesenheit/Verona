import matplotlib as mpl
from matplotlib import rc

rc("text", usetex=True)
rc("font", family="serif", size=12)
mpl.rcParams["text.latex.preamble"] = r"\usepackage[T1]{fontenc}\usepackage{lmodern}"
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["axes.titlesize"] = 12
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["legend.fontsize"] = 12
mpl.rcParams["font.size"] = 12

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import srrp

def load_verona_dump(fname: str, gamma: float):
    with h5py.File(fname, "r") as f:
        data = f["data"][()]
        T = float(f["T"][()]) if "T" in f else None
        grid = f["grid"][()] if "grid" in f else None

    if data.ndim != 4 or data.shape[-1] != 5:
        raise RuntimeError(f"Nieoczekiwany kształt data: {data.shape}")

    Nz, Ny, Nx, _ = data.shape
    z = Nz // 2
    y = Ny // 2
    line = data[z, y, :, :]

    rho = line[:, 0]
    vx  = line[:, 1]
    u   = line[:, 4]
    p   = rho * u * (gamma - 1.0)

    if grid is not None and len(grid) >= 1:
        dx = float(grid[0])
        x = (np.arange(Nx) + 0.5) * dx
    else:
        x = np.arange(Nx, dtype=float)

    return x, rho, vx, p, u, T

def exact_shocktube_I(x: np.ndarray, t: float, x0: float, gamma: float):
    solver = srrp.Solver()
    stateL = srrp.State(rho=1.0, pressure=1,  vx=0.9, vt=0.0)
    stateR = srrp.State(rho=1.0, pressure=10, vx=0.0, vt=0.0)
    sol = solver.solve(stateL, stateR, gamma)
    xi = (x - x0) / t
    st = sol.getState(xi)
    return st.rho, st.vx, st.pressure

def main(dump_fname: str):
    gamma = 4.0 / 3.0
    x0 = 0.5
    N = 1

    x_num, rho_num, vx_num, p_num, u_num, t_dump = load_verona_dump(dump_fname, gamma)
    if t_dump is None:
        raise RuntimeError("Brak datasetu 'T' w pliku HDF5.")

    rho_ex, vx_ex, p_ex = exact_shocktube_I(x_num, t_dump, x0, gamma)

    x_ver   = x_num[::N]
    rho_ver = rho_num[::N]
    vx_ver  = vx_num[::N]
    p_ver   = p_num[::N]

    fig, axs = plt.subplots(
        1, 3,
        figsize=(15, 5),
        constrained_layout=True
    )

    exact_kw  = dict(color="k", lw=2.0, ls="-", label=r"\textsc{Exact}")
    verona_kw = dict(color="red", ls="--", ms=5.5, mew=1.2, label=r"\textsc{Verona}")

    for ax in axs:
        ax.set_box_aspect(1)
        ax.grid(True, ls="--", alpha=0.4)

    axs[0].plot(x_num, rho_ex, **exact_kw)
    axs[0].plot(x_ver, rho_ver, **verona_kw)
    axs[0].set_xlabel(r"$x$")
    axs[0].set_ylabel(r"$\rho$")
    axs[0].legend()

    axs[1].plot(x_num, vx_ex, **exact_kw)
    axs[1].plot(x_ver, vx_ver, **verona_kw)
    axs[1].set_xlabel(r"$x$")
    axs[1].set_ylabel(r"$v_x$")
    axs[1].legend()

    axs[2].plot(x_num, p_ex, **exact_kw)
    axs[2].plot(x_ver, p_ver, **verona_kw)
    axs[2].set_xlabel(r"$x$")
    axs[2].set_ylabel(r"$p$")
    axs[2].legend()

    fig.suptitle(
        rf"{{Shock Tube III}}"
    )
    pdf_name = f"shocktube_III_t{t_dump:g}.pdf"
    fig.savefig(pdf_name, format="pdf", bbox_inches="tight")
    
  #  plt.show()

if __name__ == "__main__":
    dump_fname = sys.argv[1] if len(sys.argv) > 1 else "dump100.h5"
    main(dump_fname)
