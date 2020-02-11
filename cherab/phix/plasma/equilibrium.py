import os
import numpy as np

from raysect.core import Point2D
from cherab.tools.equilibrium.efit import EFITEquilibrium
from cherab.phix.machine.wall_outline import INNER_LIMITER


class PHiXEquilibrium:
    """Read and process TSC equilibrium data

    Parameters
    ------------
    folder : str
        folder name stored in data folder, default "phix10"
    """

    def __init__(self, folder="phix10"):
        # store data path
        self.folder = folder
        self.path = os.path.join(os.path.dirname(__file__), "data", folder)

        # load TSC data
        self._load_tsc_params()

    def create_EFIT(self):
        """Return EFIT equilibrium instance
        """
        # temporaly values
        self.x_points = []
        self.strike_points = []
        self.f_profile = self.q_profile
        self.b_vacuum_radius = 0.0
        self.b_vacuum_magnitude = 0.0
        self.lcfs_polygon = self.limiter_polygon
        self.time = 0.0

        return EFITEquilibrium(
            self.r,
            self.z,
            self.psi,
            self.psi_axis,
            self.psi_lcfs,
            self.magnetic_axis,
            self.x_points,
            self.strike_points,
            self.f_profile,
            self.q_profile,
            self.b_vacuum_radius,
            self.b_vacuum_magnitude,
            self.lcfs_polygon,
            self.limiter_polygon,
            self.time,
        )

    def _load_tsc_params(self):
        """load TSC data from .dat file
        """
        # load header information
        path_flux = os.path.join(self.path, "flux.dat")
        with open(path_flux, "r") as data:
            lines = data.read()
            lines = lines.split("\n")
        header = {}
        for line in lines[3:14]:
            lin = line.split("# ")[1].split(":")
            header[lin[0]] = lin[1]
        self.header = header

        # magnetic axis
        magnetic_axis = header["position of mgnetic axis [m]"].split()
        self.magnetic_axis = Point2D(float(magnetic_axis[1]), float(magnetic_axis[2]))

        # Ip [MA]
        self.Ip = float(header['plasma current "Ip"[MA]'])

        # flux mesh size
        mesh_N = lines[15].split()
        self.mesh_N = (int(mesh_N[1]), int(mesh_N[2]))

        # Limiter polygon
        self.limiter_polygon = INNER_LIMITER[0:-1, :].transpose()

        # r, z, psi
        data = np.loadtxt(path_flux)
        self.r = data[:: self.mesh_N[0] - 1, 0]
        self.z = data[: self.mesh_N[1], 1]
        self.psi = data[:, 2].reshape((self.mesh_N[1], -1), order="F")
        self.psi = self.psi.transpose() * -1

        # q profile & psi_lcfs, psi_axis
        path_profiles = os.path.join(self.path, "profiles.dat")
        data = np.loadtxt(path_profiles, usecols=(0, 2))
        self.psi_lcfs = data[-1, 0]
        self.psi_axis = data[0, 0]
        q_profile_psin = (data[:, 0] - self.psi_axis) / (self.psi_lcfs - self.psi_axis)
        self.q_profile = np.stack((q_profile_psin, data[:, 1]))

        # j_phi
        path_jphi = os.path.join(self.path, "jphi.dat")
        data = np.loadtxt(path_jphi)
        self.j_phi = data[:, 2].reshape((self.mesh_N[1], -1), order="F")
