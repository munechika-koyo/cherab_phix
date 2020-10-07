import os
import numpy as np
import scipy
from raysect.optical import World
from cherab.phix.plasma import TSCEquilibrium
from cherab.phix.tools import import_phix_rtm, laplacian_matrix

# ------------ obtain cherab_phix path ----------------------
DIR = os.path.abspath(__file__).split(sep="demos")[0]
# folder path storing matrices
DEF_PATH = os.path.join(DIR, "output", "data", "RTM_2020_06_16_12_30_wo_ref")
# -----------------------------------------------------------


def generate_SVD(matrix_name=None, path=DEF_PATH, laplacian=True):
    """Generate SVD components and laplacian matrix for inversion procedure
    SVDs is stored under directory storing RTM, which is named as "w_laplacian" or "wo_laplacian".

    Parameters
    ----------
    matrix_name : str
        filename of RTM matrix, by default None
    path : str, optional
        directory name storing RTM, by default ../output/data/RTM_...
    laplacian : bool, optional
        whether or not to generate Laplacian matrix, by default True
        In case it is generated, inversion_base_vectors is also created and stored.
    """

    # list componets valuables' name which is indentical to their name of valuables
    list_values_name = ["u", "s", "vh"]

    # quired variables
    world = World()
    eq = TSCEquilibrium()
    rtm = import_phix_rtm(world, eq)

    # import RTM matrix
    print(f"loading RTM {matrix_name} stored in {path}.")
    matrix_3D = np.load(os.path.join(path, matrix_name))
    matrix = matrix_3D.reshape((-1, rtm.bins))

    print("generating SVD components... ")
    # generate Laplacian matrix if laplacian==True
    if laplacian is True:
        L = laplacian_matrix(rtm.voxel_map.squeeze(), dir=8)
        L_inv = scipy.linalg.inv(L)
        list_values_name.append("L_inv")

        # SVD decomposition
        u, s, vh = scipy.linalg.svd(np.dot(matrix, L_inv), full_matrices=False)

        # generate ImageBase
        inversion_base_vectors = np.dot(L_inv, vh.T)
        list_values_name.append("inversion_base_vectors")
    else:
        u, s, vh = scipy.linalg.svd(matrix, full_matrices=False)

    # save
    # make dir under setting path dir
    print("saving...")
    if laplacian is True:
        sub_dir = "w_laplacian"
    else:
        sub_dir = "wo_laplacian"

    os.makedirs(os.path.join(path, sub_dir), exist_ok=True)

    for filename in list_values_name:
        np.save(os.path.join(path, sub_dir, filename + ".npy"), eval(filename))
        print(f"completed saving {filename}")


if __name__ == "__main__":
    RTM_dir_list = sorted(os.listdir(os.path.join(DIR, "output", "RTM")))
    RTM_dir_path = save_path = os.path.join(DIR, "output", "RTM", RTM_dir_list[-1])
    save_path = RTM_dir_path
    generate_SVD(matrix_name="RTM.npy", path=save_path, laplacian=True)
    # generate_SVD(matrix_name="RTM.npy", path=save_path, laplacian=False)
