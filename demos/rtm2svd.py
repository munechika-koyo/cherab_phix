"""
Calculate SVD with/without laplacian operator
=============================================

SVD is performed by the following equation:

.. math::

    U\\Sigma V^\\mathsf{T} = AL^{-1},

where :math:`A` is an RTM, :math:`L` is the regularization operator
(e.g. :math:`L=\\text{Laplacian}` if using Phillips regularization).

We proceed with the following process to compute SVD:

1. Compute :math:`L^{-1}:
2. Compute the product of matrices :math:`AL^{-1}`
3. Compute SVD with :math:`$AL^{-1}`

To reduce memory usage, each generated arrays are stored on disk and deleted from RAM.
Additionally, we compute the folloing matrix for the future inversion work.

4. Compute :math:`L^{-1}V`
"""
# %%
# Load modules
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from raysect.optical import World

from cherab.phix.tools import Spinner, laplacian_matrix
from cherab.phix.tools.raytransfer import import_phix_rtc

# %%
# Toggle whether or not to use laplacian matrix
USE_LAPLACIAN = True

# %%
# Path and diffinitions
# ---------------------
ROOT = Path(__file__).parent.parent
RTM_DIR = ROOT / "output" / "RTM" / "2022_12_13_00_49_29"
SAVE_DIR = RTM_DIR / "w_laplacian" if USE_LAPLACIAN else RTM_DIR / "wo_laplacian"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# %%
# Load RayTransfer Cylinder object
# --------------------------------
world = World()
rtc = import_phix_rtc(world)

# %%
# 1. Compute :math:`L^{-1}`
# -------------------------
# if `USE_LAPLACIAN` is `True`, then calculate it.
if USE_LAPLACIAN:
    with Spinner("Compute the inverse Laplacian matrix", timer=True) as sp:
        laplacian = laplacian_matrix(rtc.voxel_map, dir=8)
        L_inv = open_memmap(
            SAVE_DIR / "L_inv.npy", mode="w+", dtype=np.float64, shape=laplacian.shape
        )

        # compute L^-1
        L_inv[:] = np.linalg.inv(laplacian)
        sp.ok()

# %%
# 2. Compute the product of matrices :math:`AL^{-1}`
# --------------------------------------------------
# if `USE_LAPLACIAN` is `True`, then calculate it.
# Load RTM with reshape to 2D from 3D array
if USE_LAPLACIAN:
    rtm = open_memmap(RTM_DIR / "rtm.npy", mode="r").reshape((-1, rtc.bins))

    AL_inv = open_memmap(
        SAVE_DIR / "AL_inv.npy", mode="w+", dtype=np.float64, shape=rtm.shape
    )

    # compute AL^-1
    with Spinner("Compute the product of matrices AL^-1", timer=True) as sp:
        AL_inv[:] = np.dot(rtm, L_inv)
        sp.ok()
else:
    AL_inv = open_memmap(RTM_DIR / "rtm.npy", mode="r").reshape((-1, rtc.bins))


# %%
# 3. Compute SVD
# --------------
# define array shape number
m, n = AL_inv.shape
k = min(n, m)

u_row = k if m < n else m
vh_col = n if m < n else k

# create memory-map to store
u = open_memmap(SAVE_DIR / "u.npy", dtype=np.float64, mode="w+", shape=(u_row, k))
s = open_memmap(SAVE_DIR / "s.npy", dtype=np.float64, mode="w+", shape=(k,))
vh = open_memmap(SAVE_DIR / "vh.npy", dtype=np.float64, mode="w+", shape=(k, vh_col))

# compute SVD
with Spinner("Compute SVD", timer=True) as sp:
    u[:], s[:], vh[:] = np.linalg.svd(AL_inv, full_matrices=False)
    sp.ok()


# %%
# 4. Compute :math:`L^{-1}V`
# --------------------------
# if `USE_LAPLACIAN` is `True`, then calculate it.
if USE_LAPLACIAN:
    L_inv_V = open_memmap(
        SAVE_DIR / "L_inv_V.npy", dtype=np.float64, mode="w+", shape=(n, k)
    )
    with Spinner("Compute the product of matrices L^-1 V", timer=True) as sp:
        L_inv_V[:] = np.dot(L_inv, vh.T)
        sp.ok()
