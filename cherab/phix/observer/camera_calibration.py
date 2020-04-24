import os
import cv2
import numpy as np

# from matplotlib import pyplot

# absolute path to data directory
CARIB_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")

# manual input image Points corresponding to model points (Limiter edge points)
IMAGEPOINTS = np.array(
    [(150, 118), (153, 411), (66, 400), (28, 315), (25, 220), (62, 135), (221, 20), (226, 500)],
    dtype="double",
)


class Calibration:
    """class for camera calibration using OpenCV module.

    Parameters
    ---------
    focal_length: float, optional
        camera's focal length, by default 10mm
    pixel_pitch: float, optional
        camera's pixel pitch, by default 20$\\mu$m
    dist_coeffs: array of 4x1
        distortion coefficients 4, 5 or 8 elements,
        by default numpy.zeros(4, 1)

    Attributes
    ------------
    calibration(): function
        excute caliration algorithm by computing solvePnP probrem
        and obtain camera's extrisic parameters: rotation matrix & translation vector
    get_camera_position(): function
        get camera's position in world coordinates.
    rotationMatrixToEulerAngles(): function
        get euler angles by using rotation matrix
    """

    def __init__(self, focal_length=None, pixel_pitch=None):
        # initialization paramters
        self._focal_length = focal_length or 10.0e-3
        self._pixel_pitch = pixel_pitch or 20.0e-6
        self._rotation_matrix = None
        self._tranlation_vector = None
        self.model_points = None
        self.image_points = None
        self._camera_matrix = None
        self._dist_coeffs = np.zeros((4, 1))

    @property
    def focal_length(self):
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value):
        if value < 0:
            raise ValueError("focal_length must be greater than 0")
        self._focal_length = value

    @property
    def pixel_pitch(self):
        return self._pixel_pitch

    @pixel_pitch.setter
    def pixel_pitch(self, value):
        if value < 0:
            raise ValueError("pixel_pitch must be greater than 0")
        self._pixel_pitch = value

    @property
    def rotation_matrix(self):
        return self._rotation_matrix

    @rotation_matrix.setter
    def rotation_matrix(self, mat):
        if not isinstance(mat, type(np.array([]))):
            raise ValueError("rotaion matrix must be numpy.ndarray.")
        if mat.shape != (3, 3):
            raise ValueError("rotation matrix must be (3, 3) shape matrix.")

        self._rotation_matrix = mat

    @property
    def translation_vector(self):
        return self._translation_vector

    @translation_vector.setter
    def translation_vector(self, vector):
        if not isinstance(vector, type(np.array([]))):
            raise ValueError("translation vector must be numpy.ndarray.")
        if vector.shape != (3,):
            raise ValueError("translation vector must be (3,) shape matrix")

        self._tranlation_vector = vector

    @property
    def camera_matrix(self):
        return self._camera_matrix

    @camera_matrix.setter
    def camera_matrix(self, mat):
        raise NotImplementedError("camera matrix must be set in function calc_camera_matrix().")

    @property
    def dist_coeffs(self):
        return self._dist_coeffs

    @dist_coeffs.setter
    def dist_coeffs(self, value):
        if len(value) not in (4, 5, 8):
            raise ValueError(
                "distortion coefficients must be a vector consisting of 4, 5, or 8 elements."
            )
        self._dist_coeffs = value

    def calibrate(
        self, path=CARIB_PATH, model_points=None, image_points=None, ref_image_filename=None
    ):
        """calibration of camera and obtain position & orientation

        Parameters
        ----------
        path : str, optional
            path to the directory in order to store calibration parameters,
            by default /cherab/phix/observer/data/
        model_points: (8, 3) numpy.ndarray, optional
            Array of object points in the world coordinates
            if is None, model points is loaded from the data directory.
        image_points: (8, 2) numpy.ndarray, optional
            Array of corresponding image points in pixel unit.
            if is None, it is loaded from the data directory.
        ref_image_filename : str, optional
            reference image file name which is needed to set as absolute path,
            by default "path/shot_10722.png".
        """
        # initialization of parameters
        ref_image_filename = ref_image_filename or os.path.join(path, "shot_10722.png")
        model_points = model_points or np.loadtxt(
            os.path.join(CARIB_PATH, "LimiterPoints.csv"), delimiter=","
        )
        image_points = image_points or IMAGEPOINTS

        # import reference image file
        im = cv2.imread(ref_image_filename, cv2.IMREAD_COLOR)
        size = im.shape

        # calculation of camera matrix
        self._camera_matrix = self.calc_camera_matrix(size)

        # solve Perspective problem (PnP problem)
        (success, rotation_vector, self._translation_vector) = cv2.solvePnP(
            model_points,
            image_points,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        self._rotation_mat, _ = cv2.Rodrigues(rotation_vector)

        # display results of calibration
        print("model Points [m]:\n{0}".format(model_points))
        print("Translation Vector:\n{0}".format(self._translation_vector))
        print("Rotaion Vector:\n{0}".format(rotation_vector))
        print("Rotaion Matrix:\n{0}".format(self._rotation_mat))

        # save translation_vector, rotation_matrix csv file into path directory
        np.savetxt(os.path.join(path, "translation_vector.csv"), self._translation_vector)
        np.savetxt(os.path.join(path, "rotation_matrix.csv"), self._rotation_mat)

        # reprojection model projectPoints
        (reprojectP, jacobian) = cv2.projectPoints(
            model_points,
            rotation_vector,
            self._translation_vector,
            self._camera_matrix,
            self._dist_coeffs,
        )
        """
        with open('reprojectedPoints.csv','a') as f_handle:
            i = 0
            while i < reprojectP.shape [0]:
                np.savetxt(f_handle,reprojectP[i],delimiter=',')
                i += 1
                """

        # Axis reprojection to image
        (origin, jacobian) = cv2.projectPoints(
            np.array([(0.0, 0.0, 0.0)]),
            rotation_vector,
            self._translation_vector,
            self._camera_matrix,
            self._dist_coeffs,
        )
        (xaxis, jacobian) = cv2.projectPoints(
            np.array([(0.5, 0.0, 0.0)]),
            rotation_vector,
            self._translation_vector,
            self._camera_matrix,
            self._dist_coeffs,
        )
        (yaxis, jacobian) = cv2.projectPoints(
            np.array([(0.0, 0.5, 0.0)]),
            rotation_vector,
            self._translation_vector,
            self._camera_matrix,
            self._dist_coeffs,
        )
        (zaxis, jacobian) = cv2.projectPoints(
            np.array([(0.0, 0.0, 0.5)]),
            rotation_vector,
            self._translation_vector,
            self._camera_matrix,
            self._dist_coeffs,
        )
        origin = (int(origin[0][0][0]), int(origin[0][0][1]))
        xaxis = (int(xaxis[0][0][0]), int(xaxis[0][0][1]))
        yaxis = (int(yaxis[0][0][0]), int(yaxis[0][0][1]))
        zaxis = (int(zaxis[0][0][0]), int(zaxis[0][0][1]))

        for i, p in enumerate(image_points):
            cv2.circle(im, (0, 0), 10, (0, 0, 0), -1)  # (B,G,R)
            cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)  # 赤色は選択したイメージポイント
            cv2.circle(
                im, (int(reprojectP[i][0][0]), int(reprojectP[i][0][1])), 1, (0, 255, 0), -1
            )  # 緑点が作成したアフィン変換行列による空間座標を画像上の点に変化したもの

            # drow xyz axis
            cv2.line(im, origin, xaxis, (0, 0, 255), 2)
            cv2.line(im, origin, yaxis, (0, 255, 0), 2)
            cv2.line(im, origin, zaxis, (255, 0, 0), 2)

        # display results
        cv2.imshow("Red: selected points, Green: reprojected points", im)

    def calc_camera_matrix(self, image_size):
        """compute camera matrix

        Parameter
        ---------
        image_size: tuple
            image size tuple (w, h) in pixel unit
        """
        # transform focal length's unit to [px]
        _focal_length = self._focal_length / self._pixel_pitch
        center = (0.5 * image_size[1], 0.5 * image_size[0])
        camera_matrix = np.array(
            [[_focal_length, 0, center[0]], [0, _focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )
        return camera_matrix

    def get_camera_position(self, rotation_matrix=None, translation_vector=None):
        """Obtain camera's position in world coordinates

        Parameters
        ----------
        rotation_matrix: (3, 3) numpy.ndarray
            rotation matrix
        translation_vector: (3, ) numpy.ndarray
            translation vector

        Return
        ----------
        array([x, y, z]): (3, ) numpy.ndarray
            camera position in world coordinates
        """
        _rotation_matrix = rotation_matrix or self._rotation_matrix
        _translation_vector = translation_vector or self._tranlation_vector
        if _rotation_matrix is None or _translation_vector is None:
            raise ValueError("rotation matrix or translation vector have not been defined yet.")

        camera_position = -np.dot(_rotation_matrix.T, _translation_vector.reshape(3, 1))

        return camera_position.ravel()

    def rotationMatrixToEulerAngles(self, rotation_matrix=None):
        """Calculates rotation matrix to euler angles
        The result is the same as MATLAB except the order
        of the euler angles ( x and z are swapped ).
        check if R matrix is a valid rotation matrix

        Parameter
        ---------
        rotation_matrix: numpy.ndarray
            rotation matrix

        Returns
        --------
        numpy.ndarray
            euler angles
        """
        # initalization
        if self._rotation_matrix is None:
            rotation_matrix = rotation_matrix
        else:
            rotation_matrix = self._rotation_matrix

        if not isinstance(rotation_matrix, type(np.array([]))):
            raise ValueError("rotation matrix must be numpy array.")
        if rotation_matrix.shape != (3, 3):
            raise ValueError("rotaion matrix must be (3, 3) array.")
        assert self._isRotationMatrix(rotation_matrix)

        sy = np.sqrt(
            rotation_matrix[0, 0] * rotation_matrix[0, 0]
            + rotation_matrix[1, 0] * rotation_matrix[1, 0]
        )

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    def _isRotationMatrix(self, R):
        """Checks if a matrix is a valid rotation matrix.
        """
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        identity = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(identity - shouldBeIdentity)
        return n < 1e-6

    def calc_up_forward_vector(self, rotation_matrix=None):
        """calculation of camera's up & forward vector

        Parameter
        ----------
        rotation_matrix: (3, 3) numpy.ndarray
            rotation matrix

        Returns
        ---------
        up: (3, ) numpy.ndarray
            up vector
        forward: (3, ) numpy.ndarray
            forward vector
        """
        # initalization
        if self._rotation_matrix is None:
            rotation_matrix = rotation_matrix
        else:
            rotation_matrix = self._rotation_matrix

        if not isinstance(rotation_matrix, type(np.array([]))):
            raise ValueError("rotation matrix must be numpy array.")
        if rotation_matrix.shape != (3, 3):
            raise ValueError("rotaion matrix must be (3, 3) array.")

        up = np.dot(rotation_matrix.T, np.array([[0], [-1], [0]]))
        forward = np.dot(rotation_matrix.T, np.array([[0], [0], [1]]))

        return up.ravel(), forward.ravel()


if __name__ == "__main__":
    calib = Calibration()
    calib.calibrate()
