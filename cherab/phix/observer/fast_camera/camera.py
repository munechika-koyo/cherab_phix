import os
import numpy as np
from raysect.optical import translate, AffineMatrix3D

# from raysect.optical.observer import TargettedCCDArray

from cherab.phix.observer import ThinLensCCDArray

# from raysect.primitive.lens.spherical import BiConvex
# from raysect.optical.material import NullMaterial
# from raysect.optical.library import schott


# path to direction storing camera's extinction parameters
CARIB_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "calibration_data")


def import_phix_camera(paret, path=None):
    """importing phix lens camera set as defalut camera parameters

    Parameters
    ----------
    parent : Node
        Raysect's scene-graph paret node
    path : str, optional
        absolute path to the directory storing camera's extinction parameters,
        by default "../cherab/phix/observer/data"

    Returns
    --------
    object
        cherab.phix's ThinLensCCDArray object
    """
    print("importing PHiX camera...")
    # initialise parameters
    path = path or CARIB_PATH

    # import camera extinct parameters
    translation_vector = np.loadtxt(os.path.join(path, "translation_vector.csv"))
    rotation_mat = np.loadtxt(os.path.join(path, "rotation_matrix.csv"))

    orientation = rotation_mat.T
    camera_pos = -np.dot(orientation, translation_vector.reshape((3, 1)))
    camera_trans = translate(*camera_pos)
    camera_rot = np.block([[orientation, np.zeros((3, 1))], [np.array([0, 0, 0, 1])]])
    camera_rot = AffineMatrix3D(camera_rot)
    # generate ThinLensCCDArray object
    camera = ThinLensCCDArray(
        pixels=(256, 512),
        width=25.6e-3 * 256 / 1280,
        focal_length=1.0e-2,
        working_distance=50.0e-2,
        F_value=0 * (22 - 3.5) / 10 + 3.5,
        parent=paret,
        pipelines=None,
        transform=camera_trans * camera_rot,
        name="PHiX fast-visible camera",
    )

    return camera


# obstacle
# class LensCamera(TargettedCCDArray):
#     """Lens Camera using thin lens model inherited TargettedCCDArray
#     This class uses a BiConvex lens and put it on the origin in local axis.
#     The sensor is located behind the Lens on the z-axis. The distance from lens
#     is calculated by Lens equation using the focal length and Working distance.

#     Parameters
#     ----------
#     pixels : tuple, optional
#         A tuple of pixel dimensions for the camera, by default (256, 256)
#     width : float, optional
#         The CCD sensor x-width in metres, by default 35mm
#     focal_length : float, optional
#         focal length in metres, by default 10mm
#     working_distance : float, optional
#         working distance from lens to focus plane in metres, by default 50.0cm
#     F_value : float, optional
#         F value, by default 3.5
#     pipelines : pipelines, optional
#         The list of pipelines that will process the spectrum measured
#         at each pixel by the camera, by default RGBPipeline2D()
#     kwargs :
#         **kwargs and properties from Observer2D and _ObserverBase.
#     """

#     def __init__(
#         self,
#         pixels=(256, 256),
#         width=0.035,
#         focal_length=10e-3,
#         working_distance=50.0e-2,
#         F_value=3.5,
#         parent=None,
#         transform=None,
#         name=None,
#         pipelines=None,
#     ):
#         # camera base Node
#         camera = Node(parent=parent, transform=transform)

#         # initial values to prevent undefined behaviour
#         self._focal_length = focal_length
#         self._F_value = F_value
#         self._working_distance = working_distance

#         # Lens (material: N-BK7)
#         material = schott("N-BK7")
#         material.transmission_only = True
#         diamiter = self._focal_length / self._F_value
#         # calculate curvature by using thin lens model
#         curvature = 2 * (material.index.sample(375, 780, 10).mean() - 1) * self._focal_length
#         center_thickness = 2 * (curvature - np.sqrt(curvature ** 2 - (diamiter / 2) ** 2))
#         # lens = PlanoConvex(
#         lens = BiConvex(
#             diamiter,
#             center_thickness,
#             curvature,  # The radius of curvature of the spherical front surface.
#             curvature,  # The radius of curvature of the spherical front surface.
#             parent=camera,
#             transform=translate(0, 0, -center_thickness / 2),
#             material=material,
#         )
#         # aperture
#         # aperture_radius = 0.5 * self._focal_length / self._F_value
#         # aperture_thin = 0.0009
#         # aperture = Cylinder(
#         #     aperture_radius,
#         #     aperture_thin,
#         #     parent=camera,
#         #     transform=None,
#         #     material=NullMaterial(),
#         #     name="aperture",
#         # )
#         # lens equation
#         lens_to_image = 1 / ((1 / self._focal_length) - (1 / self._working_distance))
#         super().__init__(
#             [lens],
#             pixels=pixels,
#             width=width,
#             parent=camera,
#             targetted_path_prob=1.0,
#             transform=translate(0, 0, -1 * lens_to_image) * rotate_z(180),
#             name=name,
#             pipelines=pipelines,
#         )

#         # # save setting parameters
#         # self.focal_length = focal_length
#         # self.F_value = F_value
#         # self.working_distance = working_distance

#     @property
#     def focal_length(self):
#         return self._focal_length

#     @focal_length.setter
#     def focal_length(self, value):
#         focal_lenth = value
#         if focal_lenth <= 0:
#             raise ValueError("Focal length must be greater than 0.")
#         self._focal_length = focal_lenth

#     @property
#     def F_value(self):
#         return self._F_value

#     @F_value.setter
#     def F_value(self, value):
#         F_value = value
#         if F_value <= 0:
#             raise ValueError("F value must be greater than 0.")
#         self._F_value = F_value

#     @property
#     def working_distance(self):
#         return self._working_distance

#     @working_distance.setter
#     def working_distance(self, value):
#         working_distance = value
#         if working_distance <= 0:
#             raise ValueError("Working distance must be greater than 0.")
#         self._working_distance = working_distance