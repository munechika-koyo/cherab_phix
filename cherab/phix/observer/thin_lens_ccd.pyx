"""Module to offer the thin lens modeled observer object
"""
from __future__ import annotations

from raysect.optical.observer.pipeline import RGBPipeline2D
from raysect.optical.observer.sampler2d import FullFrameSampler2D

cimport cython
from libc.math cimport M_PI
from raysect.core.math.sampler cimport DiskSampler3D, RectangleSampler3D
from raysect.optical cimport AffineMatrix3D, Point3D, Ray, Vector3D, translate
from raysect.optical.observer.base cimport Observer2D

__all__ = ["ThinLensCCDArray"]


cdef class ThinLensCCDArray(Observer2D):
    """
    An ideal CCD-like imaging sensor that preferentially targets a thin lens circle.

    The targetted CCD is a regular array of square pixels. Each pixel samples red, green
    and blue channels (behaves like a Foveon imaging sensor). The CCD sensor
    width is specified with the width parameter. The CCD height is calculated
    from the width and the number of vertical and horizontal pixels. The
    default width and sensor ratio approximates a 35mm camera sensor.

    Each pixel will target the randomly sampled point inside the circle which is modeled as a thin lens.
    The lens radius is caluclated by the f-number and focal length.
    The number of samples is pixel_samples multipled by lens_samples, so total number of sampling rays
    is taken as a pixel_samples in Observer2D object.

    Parameters
    ----------
    pixels : tuple
        A tuple of pixel dimensions for the camera (default=(512, 512)).
    width : float
        The CCD sensor x-width in metres (default=35mm).
    focal_length : float
        The focal length in metres (default=10mm).
    working_distance : float
        The working distance from lens to focus plane in metres (default=50.0cm).
    f_number : float
        f-number (default=3.5).
    lens_samples : int
        Number of samples to generate on the thin Lens (default=100).
    per_pixel_samples : int
        Number of samples to generate per pixel (default=10).
        which is different from `pixel_samples` in :py:class:`~raysect.optical.observer.base.observer.Observer2D` object.
        When you use ThinLensCCDArray, `pixel_samples` = `per_pixel_samples` x `lens_samples` (default=100 x 10).
    pipelines : list
        The list of pipelines that will process the spectrum measured
        at each pixel by the camera (default= :py:class:`~raysect.optical.observer.pipeline.rgb.RGBPipeline2D`).
    **kwargs : :py:class:`~raysect.optical.observer.base.observer.Observer2D` and _ObserverBase. properties, optional
        *kwargs* are used to specify properties like a parent, transform, pipelines, etc.
    """

    cdef:
        int _lens_samples, _per_pixel_samples
        double _width, _pixel_area, _image_delta, _image_start_x, _image_start_y
        double _focal_length, _working_distance, _f_number, _lens_radius, _lens_area
        double _image_distance
        RectangleSampler3D _pixel_sampler
        DiskSampler3D _lens_sampler

    def __init__(self, pixels=(720, 480), width=0.035, focal_length=10.e-3,
                 working_distance=50.e-2, f_number=3.5, lens_samples=100, per_pixel_samples=10,
                 parent=None, transform=None, name=None, pipelines=None):

        # initial values to prevent undefined behaviour when setting via self.width
        self._width = 0.035
        self._pixels = (720, 480)
        self._focal_length = focal_length
        self._working_distance = working_distance
        self._f_number = f_number
        self._lens_samples = lens_samples
        self._per_pixel_samples = per_pixel_samples

        pipelines = pipelines or [RGBPipeline2D()]

        super().__init__(pixels, FullFrameSampler2D(), pipelines,
                         pixel_samples=self._lens_samples * self._per_pixel_samples,
                         parent=parent, transform=transform, name=name)

        # setting width & focal_length trigger calculation of image geometry &
        # lens geometry calculations, respectively.
        self.width = width
        self.focal_length = focal_length

    @property
    def pixels(self) -> tuple[int, int]:
        """
        Tuple describing the pixel dimensions for this observer (nx, ny), i.e. (512, 512).

        :rtype: tuple[int, int]
        """
        return self._pixels

    @pixels.setter
    def pixels(self, value):
        pixels = tuple(value)
        if len(pixels) != 2:
            raise ValueError("Pixels must be a 2 element tuple defining the x and y resolution.")
        x, y = pixels
        if x <= 0:
            raise ValueError("Number of x pixels must be greater than 0.")
        if y <= 0:
            raise ValueError("Number of y pixels must be greater than 0.")
        self._pixels = pixels
        self._update_image_geometry()

    @property
    def width(self) -> float:
        """
        The CCD sensor x-width in metres.

        :rtype: float
        """
        return self._width

    @width.setter
    def width(self, width):
        if width <= 0:
            raise ValueError("width can not be less than or equal to 0 meters.")
        self._width = width
        self._update_image_geometry()

    @property
    def pixel_area(self) -> float:
        """
        One pixel area in the CCD sensor

        :rtype: float
        """
        return self._pixel_area

    @property
    def focal_length(self) -> float:
        """
        Focal length in metres.

        :rtype: float
        """
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value):
        focal_length = value
        if focal_length <= 0:
            raise ValueError("Focal length must be greater than 0.")
        self._focal_length = focal_length
        self._update_lens_geometry()

    @property
    def f_number(self) -> float:
        """
        f-number which defines the lens radius with focal length.

        :rtype: float
        """
        return self._f_number

    @f_number.setter
    def f_number(self, value):
        f_number = value
        if f_number <= 0:
            raise ValueError("F value must be greater than 0.")
        self._f_number = f_number
        self._update_lens_geometry()

    @property
    def working_distance(self) -> float:
        """
        distance between the lens plane and focusing plane in metres.

        :rtype: float
        """
        return self._working_distance

    @working_distance.setter
    def working_distance(self, value):
        working_distance = value
        if working_distance <= 0:
            raise ValueError("Working distance must be greater than 0.")
        self._working_distance = working_distance
        self._update_lens_geometry()

    @property
    def lens_radias(self) -> float:
        """
        Lens Radius in metres.

        :rtype: float
        """
        return self._lens_radius

    @property
    def lens_samples(self) -> int:
        """
        The number of samples on lens.

        :rtype: int
        """
        return self._lens_samples

    @lens_samples.setter
    def lens_samples(self, value):
        if value <= 0:
            raise ValueError("The number of lens samples must be greater than 0.")
        self._lens_samples = value
        self._update_lens_geometry()
        self._update_pixel_samples()

    @property
    def per_pixel_samples(self) -> int:
        """
        The number of samples to take per pixel.

        :rtype: int
        """
        return self._per_pixel_samples

    @per_pixel_samples.setter
    def per_pixel_samples(self, value):
        if value <= 0:
            raise ValueError("The number of pixel samples must be greater than 0.")
        self._per_pixel_samples = value
        self._update_pixel_samples()

    cdef object _update_image_geometry(self):

        self._image_delta = self._width / self._pixels[0]
        self._image_start_x = 0.5 * self._pixels[0] * self._image_delta
        self._image_start_y = 0.5 * self._pixels[1] * self._image_delta
        self._pixel_sampler = RectangleSampler3D(self._image_delta, self._image_delta)
        self._pixel_area = (self._width / self._pixels[0])**2

    cdef object _update_lens_geometry(self):

        self._lens_radius = 0.5 * self._focal_length / self._f_number
        self._image_distance = 1 / (1 / self._focal_length - 1 / self._working_distance)
        self._lens_sampler = DiskSampler3D(self._lens_radius)
        self._lens_area = M_PI * self._lens_radius**2

    cdef object _update_pixel_samples(self):
        # pixel_samples means total pixel samples in this case.
        self.pixel_samples = self._per_pixel_samples * self._lens_samples

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef list _generate_rays(self, int ix, int iy, Ray template, int ray_samples):
        """
        Generate a list of Rays that sample over the sensitivity of the pixel.
        This method must return a list of tuples, with each tuple containing a Ray object and
        a corresponding weighting, typically the projected area/direction cosine.
        In this class, the weight will be:

        .. math::

            wight := \\frac{R^2\\cos^4\\theta}{2d^2},

        where :math:`R` is the lens radius, :math:`\\theta` is the angle between a ray vector and
        the normal vector to image sensor, and :math:`d` is the distance between the image sensor and the lens.
        :math:`R` and :math:`d` are calculated as folows:

        .. math::

                R &= \\frac{f}{F}

                d &= \\left(\\frac{1}{f} - \\frac{1}{W}\\right)^{-1}.

        Where :math:`f` is the focal length, :math:`F` is the f-number, and :math:`W` is the working distance.

        Parameters
        ----------
        x : int
            Pixel x index.
        y : int
            Pixel y index.
        template : :obj:`~raysect.optical.ray.Ray`
            The template ray from which all rays should be generated.
        ray_count : int
            The number of rays to be generated.

        Return
        ------
        list[tuple[:obj:`~raysect.optical.ray.Ray`, float]]
            A list of tuples of (ray, weight)
        """
        cdef:
            double pixel_x, pixel_y, weight
            list pixel_origins, lens_origins, rays
            Point3D pixel_origin, lens_origin
            Vector3D pixel_direction, direction
            AffineMatrix3D pixel_to_local

        # generate pixel transform
        pixel_x = self._image_start_x - self._image_delta * ix
        pixel_y = self._image_start_y - self._image_delta * iy
        pixel_to_local = translate(pixel_x, pixel_y, -1 * self._image_distance)

        # generate origin points in pixel space
        pixel_origins = self._pixel_sampler.samples(self._per_pixel_samples)

        # assemble rays
        rays = []
        for pixel_origin in pixel_origins:

            # transform to local space from pixel space
            pixel_origin = pixel_origin.transform(pixel_to_local)

            # generate origin points in lens space (which is equal to local space)
            lens_origins = self._lens_sampler.samples(self._lens_samples)

            for lens_origin in lens_origins:

                # generate direction from a sampled pixel point to a sampled lens point
                pixel_direction = pixel_origin.vector_to(lens_origin)
                pixel_direction = pixel_direction.normalise()

                # generate ray direction from a sampled lens point (lens_origin)
                direction = pixel_origin.vector_to(Point3D(0, 0, 0)).normalise()
                direction = lens_origin.vector_to(Point3D(0, 0, 0)) + direction * self._working_distance / direction.z
                direction = direction.normalise()

                # weight = 0.5 * lens_radias^2 * cos(theta)^4 * 1/image_distance^2
                weight = 0.5 * self._lens_radius**2 * pixel_direction.z**4 / self._image_distance**2

                rays.append((template.copy(lens_origin, direction), weight))

        return rays

    cpdef double _pixel_sensitivity(self, int x, int y):
        return self._pixel_area * 2 * M_PI
