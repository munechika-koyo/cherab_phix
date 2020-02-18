import os
import numpy as np

from raysect.core import Point2D
from raysect.optical import Vector3D
from cherab.core.math import Interpolate1DCubic, Interpolate2DCubic, ClampOutput2D
from cherab.core.math import ConstantVector2D, VectorFunction2D
from cherab.tools.equilibrium.efit import EFITEquilibrium
from cherab.phix.machine.wall_outline import INNER_LIMITER


class TSCEquilibrium(EFITEquilibrium):
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
    r_data : array
        TSC grid radius axis values.
    z_data : array
        TSC grid height axis values.
    psi_data : array
        TSC psi grid values :math:`N_r \\times N_z`.
    j_phi_data : array
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
    q0 : float
        The q value at the magnetic axis.
    x_points
        A list or tuple of x-points
    f_profile
        The current flux profile on psin (2xN array).
    q_profile
        The safety factor (q) profile on psin (2xN array).
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

        # load TSC data:
        self._load_tsc_params()

        # interpolate poloidal flux grid data
        self.psi = Interpolate2DCubic(self.r_data, self.z_data, self.psi_data)
        # self.psi_axis = self.psi(self.magnetic_axis)
        # self.psi_lcfs = self._calc_psi_lcfs()
        self.psi_normalised = ClampOutput2D(
            Interpolate2DCubic(
                self.r_data,
                self.z_data,
                (self.psi_data - self.psi_axis) / (self.psi_lcfs - self.psi_axis),
            ),
            min=0,
        )
        # interpolate current flux grid data (g function)
        self.g = Interpolate2DCubic(self.r_data, self.z_data, self.g_data)

        # store equilibrium attributes
        self.r_range = self.r_data.min(), self.r_data.max()
        self.z_range = self.z_data.min(), self.z_data.max()
        self.q = Interpolate1DCubic(self.q_profile[0, :], self.q_profile[1, :])

        # populate points
        # super()._process_points(self._magnetic_axis, self._x_point, [])

        # populate polygons and inside/outside functions
        limiter_polygon = INNER_LIMITER[0:-1, :].transpose()
        super()._process_polygons(self._lcfs_polygon, limiter_polygon, self.psi_normalised)

        # calculate b-field
        dpsi_dr, dpsi_dz = super()._calculate_differentials(self.r_data, self.z_data, self.psi_data)
        self.b_field = MagneticField(dpsi_dr, dpsi_dz, self.g)

        # populate flux coordinate attributes
        self.toroidal_vector = ConstantVector2D(Vector3D(0, 1, 0))
        # self.poloidal_vector = PoloidalFieldVector(self.b_field)
        # self.surface_normal = FluxSurfaceNormal(self.b_field)

        # generate interpolator to map from psi normalised to outboard major radius
        super()._generate_psin_to_r_mapping()

    def _load_tsc_params(self):
        """load TSC data from .dat file
        """
        # ------ load miscellaneous constances. -------------------------------
        path_flux = os.path.join(self.path, "flux.dat")
        with open(path_flux, "r") as data:
            lines = data.read()
            lines = lines.split("#datatype")[0]
            lines = lines.split("\n")[0:-1]
        header = {}
        for line in lines[3:-2]:
            lin = line.split("# ")[1].split(":")
            header[lin[0]] = lin[1]
        self.header = header

        # magnetic axis
        self._magnetic_axis = eval(header["position of mgnetic axis [m]"])
        self.magnetic_axis = Point2D(self._magnetic_axis[0], self._magnetic_axis[1])

        # X-point
        self._x_point = eval(header["position of x-point [m]"])
        self.x_point = Point2D(self._x_point[0], self._x_point[1])

        # Ip [MA]
        self.Ip = float(header['plasma current "Ip"[MA]'])

        # q value at magnetic axis
        self.q0 = float(header['safety factor at magnetic axis "q0"'])

        # psi at magnetic axis (psi_axis)
        self.psi_axis = float(header['poloidal flux at magnetic axis "psi0"'])

        # psi at LCFS (psi_lcfs)
        self.psi_lcfs = float(header['poloidal flux at LCFS "psi_LCFS"'])

        # flux mesh size
        mesh_N = lines[-1].split()
        self.mesh_N = (int(mesh_N[1]), int(mesh_N[2]))

        # ---------------------------------------------------------------------
        # r, z, psi data
        data = np.loadtxt(path_flux, dtype=np.float64)
        self.r_data = data[:: self.mesh_N[0] - 1, 0]
        self.z_data = data[: self.mesh_N[1], 1]
        self.psi_data = data[:, 2].reshape((self.mesh_N[1], -1), order="F")
        self.psi_data = self.psi_data.transpose()

        # q profile
        data = np.loadtxt(os.path.join(self.path, "q.dat"), dtype=np.float64)
        q_profile_psin = (data[:, 0] - self.psi_axis) / (self.psi_lcfs - self.psi_axis)
        self.q_profile = np.stack((q_profile_psin, data[:, 1]))

        # lcfs polygon
        self._lcfs_polygon = np.loadtxt(
            os.path.join(self.path, "lcfs.dat"), dtype=np.float64
        ).transpose()

        # g function data
        data = np.loadtxt(os.path.join(self.path, "g.dat"), dtype=np.float64)
        self.g_data = data[:, 2].reshape((self.mesh_N[1], -1), order="F")
        self.g_data = self.g_data.transpose()


class MagneticField(VectorFunction2D):
    """
    A 2D magnetic field vector function derived from TSC data.

    :param dpsi_dr: A 2D function of the radius differential of poloidal flux.
    :param dpsi_dz: A 2D function of the height differential of poloidal flux.
    :param g_func: A 1D function containing a current flux profile.
    """

    def __init__(self, dpsi_dr, dpsi_dz, g_profile):

        self._dpsi_dr = dpsi_dr
        self._dpsi_dz = dpsi_dz
        self._g_profile = g_profile

    # @cython.cdivision(True)
    def __call__(self, r, z):

        # calculate poloidal components of magnetic field
        br = -self._dpsi_dz(r, z) / r
        bz = self._dpsi_dr(r, z) / r
        bt = self._g_profile(r, z) / r

        return Vector3D(br, bt, bz)
