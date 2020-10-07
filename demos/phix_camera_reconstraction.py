import os
import numpy as np

# from scipy.io import loadmat
import cv2
from raysect.optical import World
from cherab.phix.plasma import TSCEquilibrium
from cherab.phix.inversion import Lcurve
from cherab.phix.tools import profile_1D_to_2D, import_phix_rtm


# ------------------- PATH ------------------------
CHERAB_PHIX_DIR = os.path.dirname(__file__).split("demos")[0]

CAMERA_DATA_PATH = os.path.join(CHERAB_PHIX_DIR, "output/data/experiment/shot_17393/camera_data")
FRAME_DIR = os.path.join(CHERAB_PHIX_DIR, "output/data/experiment/shot_17393/camera_data/Halpha")
RECONST_DIR = os.path.join(CHERAB_PHIX_DIR, "output/data/experiment/shot_17393/camera_data/reconstraction_halpha")

# -------------- Raysect & Cherab init -----------------
world = World()
eq = TSCEquilibrium(folder="phix10")
rtm = import_phix_rtm(world, eq)
# ------------------------------------------------------

# frames list
list_frame = os.listdir(FRAME_DIR)

# load SVD components of RTM
print("load SVD components")
s = np.load(os.path.join(CHERAB_PHIX_DIR, "output", "data/RTM/2020_07_23_16_21_46/w_laplacian/s.npy"))
u = np.load(os.path.join(CHERAB_PHIX_DIR, "output", "data/RTM/2020_07_23_16_21_46/w_laplacian/u.npy"))
inverted_base = np.load(
    os.path.join(CHERAB_PHIX_DIR, "output", "data/RTM/2020_07_23_16_21_46/w_laplacian/inversion_base_vectors.npy")
)
# Generate LcuveMethod instanse
lcurv = Lcurve(s=s, u=u, inversion_base_vectors=inverted_base)
lcurv.lambdas = 10.0 ** np.linspace(-24, -14, 100)

# buffer for some information
lambda_opts = []
residual_norms = []
regu_norms = []

print("start reconstraction")
# frames iteration
for frame_name in list_frame:
    # load .npy frame file
    image = np.load(os.path.join(FRAME_DIR, frame_name))
    # resize image to half size
    image = cv2.resize(image, (int(image.shape[1] * 0.5), int(image.shape[0] * 0.5)), interpolation=cv2.INTER_AREA)

    # image = image[:, :, 0]
    # toranspose image hight to width
    # image = image.T
    # 2D -> 1D image
    image_1D = image.ravel()

    # reconstraction
    lcurv.data = image_1D
    sol_1D = lcurv.optimized_solution(itemax=5)

    # solution vector -> 2D profile in r-z plane
    sol_2D = profile_1D_to_2D(sol_1D, rtm)

    # save solution 2D
    np.save(os.path.join(RECONST_DIR, frame_name.split(".")[0]), sol_2D)

    # store some values into buffers
    lambda_opts.append(lcurv.lambda_opt)
    residual_norms.append(lcurv.residual_norm(lcurv.lambda_opt))
    regu_norms.append(lcurv.regularization_norm(lcurv.lambda_opt))


# write other information to file
with open(os.path.join(RECONST_DIR, "result.txt"), "a") as f:
    f.write("frame number, optimized regularization parameter, residual norm, regularization norm\n")
    for i, frame_name in enumerate(list_frame):
        f.write(
            "{}, {:.4e}, {:.4e}, {:.4e}\n".format(
                frame_name.split(".")[0], lambda_opts[i], residual_norms[i], regu_norms[i]
            )
        )
