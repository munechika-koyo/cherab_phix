import json
from pathlib import Path

import numpy as np

from cherab.phix.machine.wall_outline import INNER_LIMITER

BASE_PATH = Path(__file__).parent.resolve()


def convert_raw_data_to_json(path=BASE_PATH / "phix10"):
    """Generate json file data from raw data calculated TSC. output json file
    is stored in data directory as the same file name as the folder name.

    Parameters
    ----------
    path
        directory path name where data is stored, by default "phix10"
    """
    # validate argument
    if isinstance(path, (str, Path)):
        path = Path(path)
    else:
        raise TypeError("1st argument must be a type of string or pathlib.Path instance.")
    # define storing equiribrium data
    eq_data = {}

    # parse flux.dat
    path_flux = path / "flux.dat"
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
    eq_data["r"] = flux_data[:: mesh_N[0] - 1, 0].tolist()
    eq_data["z"] = flux_data[: mesh_N[1], 1].tolist()
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
        eq_data["strike_points"] = [[0, 0], [0, 0]]

    # psi at magnetic axis (psi_axis)
    psi_axis = float(header['poloidal flux at magnetic axis "psi0"'])
    eq_data["psi_axis"] = psi_axis

    # psi at LCFS (psi_lcfs)
    psi_lcfs = float(header['poloidal flux at LCFS "psi_LCFS"'])
    eq_data["psi_lcfs"] = psi_lcfs

    # q profile from q.dat file
    q_data = np.loadtxt(path / "q.dat", dtype=np.float64)
    psi_normalized = (q_data[:, 0] - psi_axis) / (psi_lcfs - psi_axis)
    eq_data["q_profile"] = np.stack((psi_normalized, q_data[:, 1])).tolist()

    # f function data fron g.dat file
    # TODO: not implement yet, tentative values are applied here
    eq_data["f_profile"] = np.stack((psi_normalized, [1] * len(psi_normalized))).tolist()

    # b_vacuum_radius & b_vacuum_magnitude
    # TODO: not implement yet
    eq_data["b_vacuum_radius"] = 1.0
    eq_data["b_vacuum_magnitude"] = 1.0

    # lcfs polygon from lcfs.dat file
    eq_data["lcfs_polygon"] = np.loadtxt(path / "lcfs.dat", dtype=np.float64).transpose().tolist()

    # time
    eq_data["time"] = 0.0

    # limitter polygon
    eq_data["limiter_polygon"] = INNER_LIMITER[0:-1, :].transpose().tolist()

    # ----------------------------------------------------------------------- #
    # ouput as json file
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(eq_data, f, indent=4)

    print(f"{path.stem} conversion completed.")


if __name__ == "__main__":
    # debug
    for i in [10, 12, 13, 14]:
        convert_raw_data_to_json(BASE_PATH / f"phix{i}")
