import os
import time
import numpy as np

from raysect.optical import World
from cherab.tools.inversions import (
    invert_sart,
    invert_constrained_sart,
    invert_regularised_lstsq,
    SartOpencl,
)
from cherab.phix.plasma import import_plasma
from cherab.phix.tools import import_phix_rtm, profile_1D_to_2D, laplacian_matrix
import datetime


# ------------ obtain cherab_phix path ----------------------
DIR = os.path.abspath(__file__).split(sep="demos")[0]
# -----------------------------------------------------------


# generate scene world
world = World()
# import phix plasma model and equilibrium
plasma, eq = import_plasma(world)
# import RTM for phix
rtm = import_phix_rtm(world, equilibrium=eq, grid_size=2.0e-3)


# import exact measured data (power) & projection matrix
measured_data_path = os.path.join(
    DIR, "output", "synthetic_data", "phix_camera(2020_06_18_01_05_40)_Ha_power.npy"
)
matrix_path = os.path.join(
    DIR, "output", "RTM_2020_05_26", "thin_lens_camera(2020_05_26_10_57_03)_RTM.npy"
)

power = np.load(measured_data_path).ravel()
matrix = np.load(matrix_path).reshape((-1, rtm.bins))

# --------------------------------------------------------------------
#                  excute inversion transform
# --------------------------------------------------------------------
start_time = time.time()
print("start excuting inversion method")

# CPU-based SART inversion
beta_laplace = None
# inv_method = "SART_wo_laplace"
# solution, convergence = invert_sart(matrix, power)

# generate laplacian generalization
laplacian_mat = laplacian_matrix(rtm.voxel_map, dir=8)
beta_laplace = 0.005
inv_method = "SART_w_laplace"
solution, convergence = invert_constrained_sart(
    matrix, laplacian_mat, power, beta_laplace=beta_laplace
)

# GPU-based SART inversion (OpenCL)
# inv_method = "SART_OpenCL"
# with SartOpencl(
#     matrix,
#     laplacian_matrix=laplacian_mat,
#     device=None,
#     block_size=1024,
#     copy_column_major=True,
#     block_size_row_maj=64,
#     use_atomic=True,
#     steps_per_thread=64,
# ) as invert_sart:
#     solution, convergence = invert_sart(power, beta_laplace=0.01)

end_time = time.time()
runtime = end_time - start_time
# ---------------------------------------------------------------------

# save
time_now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
save_dir = os.path.join(DIR, "output", "reconstruction", time_now)
# generate directry to store the data
os.makedirs(save_dir)

# save inverted solution vector
np.save(os.path.join(save_dir, "inverted_solution_vector"), solution)
# save convergence
np.save(os.path.join(save_dir, "convergence"), convergence)

# record infromation of inversion process
with open(os.path.join(save_dir, "inversion_result.txt"), "w") as f:
    result = (
        "--------------------------------------------------------------------------------\n"
        f"date            : {time_now}\n"
        f"geometry matirix: {matrix_path}\n"
        f"measured data   : {measured_data_path}\n"
        f"inversion method: {inv_method}\n"
        f"runtime         : {end_time - start_time}\n"
        f"beta_laplace    : {beta_laplace}\n"
        "--------------------------------------------------------------------------------\n"
    )
    f.write(result)
print(f"successfully end {inv_method} (runtime: {end_time - start_time})")
