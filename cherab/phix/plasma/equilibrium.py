import os
import numpy as np

from raysect.core import Point2D
from cherab.core.math import Interpolate1DCubic, Interpolate2DCubic, ClampOutput2D
from cherab.tools.equilibrium.efit import EFITEquilibrium
from cherab.phix.machine.wall_outline import INNER_LIMITER


class TSCEquilibrium:
    """Read and process Tokamak Simulation Code equilibrium data

    Parameters
    ------------
    folder : str
        folder name stored in data folder, default "phix10"

    Attributes
    --------------
    path : str
        The path address to TSC data folder.
    header : dict
        header information of TSC output data
    r : array
        TSC grid radius axis values.
    z : array
        TSC grid height axis values.
    psi_data : array
        TSC psi grid values :math:`N_r \\times N_z`.
    j_phi : array
        TSC j grid values :math:`N_r \\times N_z`.
    mesh_N : tuple
        TSC grid size :math:`(N_z, N_r)`.
    psi_axis : float
        The psi value at the magnetic axis
    psi_lcfs : float
        The psi value at the LCFS
    magnetic_axis : Point2D
        The coordinates of the magnetic axis.
    Ip : float
        The plasma current value
    x_points
        A list or tuple of x-points
    strike_points
        A list or tuple of strike-points
    f_profile
        The current flux profile on psin (2xN array).
    q_profile
        The safety factor (q) profile on psin (2xN array).
    b_vacuum_radius : float
        Vacuum B-field reference radius (in meters).
    b_vacuum_magnitude : float
        Vacuum B-Field magnitude at the reference radius.
    lcfs_polygon
        A 2xN array of [[x0, …], [y0, …]] vertices specifying the LCFS boundary.
    limiter_polygon
        A 2xN array of [[x0, …], [y0, …]] vertices specifying the limiter.
    time : float
        The time stamp of the time-slice (in seconds).

    """

    def __init__(self, folder="phix10"):
        # store data path
        self.folder = folder
        self.path = os.path.join(os.path.dirname(__file__), "data", folder)

        # load TSC data
        self._load_tsc_params()

        # interpolate poloidal flux grid data
        self.psi = Interpolate2DCubic(self.r, self.z, self.psi_data)
        self.psi_axis = self.psi(self.magnetic_axis)
        self.psi_lcfs = self._calc_psi_lcfs()
        self.psi_normalised = ClampOutput2D(
            Interpolate2DCubic(
                self.r, self.z, (self.psi_data - self.psi_axis) / (self.psi_lcfs - self.psi_axis)
            ),
            min=0,
        )

    def create_EFIT(self):
        """Return EFIT equilibrium instance
        """
        # temporaly values
        self.x_points = []
        self.strike_points = []
        self.b_vacuum_radius = 0.0
        self.b_vacuum_magnitude = 0.0
        self.lcfs_polygon = self.limiter_polygon
        self.time = 0.0

        return EFITEquilibrium(
            self.r,
            self.z,
            self.psi_data,
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
        self.psi_data = data[:, 2].reshape((self.mesh_N[1], -1), order="F")
        self.psi_data = self.psi_data.transpose()

        """
        # q profile & psi_lcfs, psi_axis
        path_profiles = os.path.join(self.path, "profiles.dat")
        data = np.loadtxt(path_profiles, usecols=(0, 2))
        self.psi_lcfs = data[-1, 0]
        self.psi_axis = data[0, 0]
        q_profile_psin = (data[:, 0] - self.psi_axis) / (self.psi_lcfs - self.psi_axis)
        self.q_profile = np.stack((q_profile_psin, data[:, 1]))
        """

        # f profile extracted from jphi
        path_jphi = os.path.join(self.path, "jphi.dat")
        data = np.loadtxt(path_jphi)
        j_phi = data[:, 2].reshape((self.mesh_N[1], -1), order="F")
        self.j_phi = j_phi.transpose()

        """
        psi_1d = self.psi.reshape(-1)
        psi_1d, sorted_index = np.unique(psi_1d, return_index=True)
        psi_1d_normalized = (psi_1d - self.psi_axis) / (self.psi_lcfs - self.psi_axis)
        self.f_profile = np.stack((psi_1d_normalized, self.j_phi.reshape(-1)[sorted_index]))
        """

    def _calc_psi_lcfs(self):
        """calculation of psi at LCFS
        """

