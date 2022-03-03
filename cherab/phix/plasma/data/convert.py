import os
import json
import numpy as np


def convert_raw_data_to_json(path="phix10"):
    """
    Generate json file data from raw data calculated TSC.
    output json file is stored in data directory as the same file name as the folder name.

    Parameters
    ----------
    path : str, optional
        path name where data is stored, by default "phix10"
    """
    # define storing equiribrium data
    eq_data = {}

    # parse flux.dat
    path_flux = os.path.join(path, "flux.dat")
    with open(path_flux, "r") as data:
        lines = data.read()
        lines = lines.split("#datatype")[0]
        lines = lines.split("\n")[0:-1]
    header = {}

    # extract header info
    for line in lines[3:-2]:
        lin = line.split("# ")[1].split(":")
        header[lin[0]] = lin[1]

    # flux mesh size
    mesh_N = lines[-1].split()
    mesh_N = (int(mesh_N[1]), int(mesh_N[2]))

    # r, z, psi
    flux_data = np.loadtxt(path_flux, dtype=np.float64)
    eq_data["r"] = flux_data[:: mesh_N[0] - 1, 0]
    eq_data["z"] = flux_data[: mesh_N[1], 1]
    psi_data = flux_data[:, 2].reshape((mesh_N[1], -1), order="F")
    eq_data["psi"] = psi_data.transpose().tolist()

    # magnetic axis
    eq_data["axis_coord"] = eval(header["position of mgnetic axis [m]"])

    # X-point
    eq_data["x_points"] = eval(header["position of x-point [m]"])

    # strike points
    if header.get("position of strike-points [m]") is not None:
        eq_data["strike_points"] = eval(header.get("position of strike-points [m]"))
    else:
        eq_data["strike_points"] = [
            [0, 0], [0, 0]
        ]

    # psi at magnetic axis (psi_axis)
    psi_axis = float(header['poloidal flux at magnetic axis "psi0"'])
    eq_data["psi_axis"] = psi_axis

    # psi at LCFS (psi_lcfs)
    psi_lcfs = float(header['poloidal flux at LCFS "psi_LCFS"'])
    eq_data["psi_lcfs"] = psi_lcfs

    # q profile from q.dat file
    q_data = np.loadtxt(os.path.join(path, "q.dat"), dtype=np.float64)
    q_profile_psin = (q_data[:, 0] - psi_axis) / (psi_lcfs - psi_axis)
    eq_data["q_profile"] = np.stack((q_profile_psin, q_data[:, 1])).tolist()

    # f function data fron g.dat file
    data = np.loadtxt(os.path.join(path, "g.dat"), dtype=np.float64)
    g_data = data[:, 2].reshape((mesh_N[1], -1), order="F")
    g_data = g_data.transpose()
    eq_data["f_profile"] = np.stack((q_profile_psin, q_data[:, 1])).tolist()

    # b_vacuum_radius & b_vacuum_magnitude
    # This values is used in EFIT but TSC does not calculate, so
    # artifical value is applied ouside lcfs.
    eq_data["b_vacuum_radius"] = 1.0
    eq_data["b_vacuum_magnitude"] = b_vacuum_magnitude  # TODO: implement how the b_vacuum is defined

    # lcfs polygon from lcfs.dat file
    eq_data["lcfs_polygon"] = np.loadtxt(
        os.path.join(path, "lcfs.dat"), dtype=np.float64
    ).transpose().tolist()

    # time
    eq_data["time"] = 0.0

    # ----------------------------------------------------------------------- #
    # ouput as json file
    with open(os.path.join(os.path.dirname(__file__), os.path.splitext(os.path.basename(path))[0])) as f:
        json.dumps(eq_data, f, indent=4)
