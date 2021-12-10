# Copyright (c) 2014-2018, Dr Alex Meakins, Raysect Project
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the Raysect Project nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from time import time
import matplotlib.pyplot as plt
import numpy as np

cimport cython
cimport numpy as np
from libc.math cimport M_PI
from raysect.optical.observer.base cimport PixelProcessor, Pipeline2D
from raysect.core.math cimport StatsArray3D, StatsArray1D
from cherab.phix.observer.fast_camera.colour cimport resample_phantom_rgb, phantom_rgb_to_srgb

from raysect.optical.spectrum cimport Spectrum

# ctypedef np.float64_t DTYPE_t

_DEFAULT_PIPELINE_NAME = "Phantom RGBPipeline"
_DISPLAY_DPI = 100
_DISPLAY_SIZE = (512 / _DISPLAY_DPI, 512 / _DISPLAY_DPI)


cdef class RGBPipeline2D(Pipeline2D):
    """
    2D pipeline of sRGB colour values.

    Converts the measured spectrum from each pixel into PhantomRGB
    colour space values. See the colour module for more
    information. The RGBPipeline2D class is the workhorse
    pipeline for visualisation of scenes with Raysect and the
    default pipeline for most 2D observers.

    :param bool display_progress: Toggles the display of live render progress
      (default=True).
    :param float display_update_time: Time in seconds between preview display
      updates (default=15 seconds).
    :param bool accumulate: Whether to accumulate samples with subsequent calls
      to observe() (default=True).
    :param float exposure_time: Phantom camera's sexposure time (default=1.0).
    :param bool auto_normalize: if True, Display values will be scaled 
      to satisfy the maximum pixel value is equal to 4095 (default=True).
    :param str name: User friendly name for this pipeline.
    """

    def __init__(self, bint display_progress=True,
                 double display_update_time=15, bint accumulate=True,
                 double exposure_time=1.0,
                 bint auto_normalize=True, str name=None):

        self.name = name or _DEFAULT_PIPELINE_NAME

        self.display_progress = display_progress
        self.display_update_time = display_update_time
        self.display_persist_figure = True

        if exposure_time <= 0:
            raise ValueError("Exposure time must be greater than 0.")

        self._exposure_time = exposure_time
        self.auto_normalize = auto_normalize

        self.accumulate = accumulate

        self.rgb_frame = None

        self._working_mean = None
        self._working_variance = None
        self._working_touched = None

        self._display_frame = None
        self._display_timer = 0
        self._display_figure = None

        self._processors = None

        self._pixels = None
        self._samples = 0

        self._quiet = False

    def __getstate__(self):

        return (
            self.name,
            self.display_progress,
            self.display_update_time,
            self.display_persist_figure,
            self._exposure_time,
            self.auto_normalize,
            self.accumulate,
            self.rgb_frame
        )

    def __setstate__(self, state):

        (
            self.name,
            self.display_progress,
            self.display_update_time,
            self.display_persist_figure,
            self._exposure_time,
            self.auto_normalize,
            self.accumulate,
            self.rgb_frame
        ) = state

        self._working_mean = None
        self._working_variance = None
        self._working_touched = None
        self._display_frame = None
        self._display_timer = 0
        self._display_figure = None
        self._pixels = None
        self._samples = 0
        self._quiet = False

    # must override automatic __reduce__ method generated by cython for the base class
    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @property
    def exposure_time(self):
        """
        The exposure time of the Phantom camera.

        :rtype: float
        """
        return self._exposure_time

    @exposure_time.setter
    def exposure_time(self, value):
        if value <= 0:
            raise ValueError("Exposure time must be greater than 0.")
        self._exposure_time = value
        self._refresh_display()

    @property
    def auto_normalize(self):
        """
        whether pixels are normalized or not. Display values will
        be scaled to satisfy the maximum pixel value is equal to 4095.

        :rtype: bool
        """
        return self._auto_normalize

    @auto_normalize.setter
    def auto_normalize(self, value):
        if not isinstance(value, bool):
            raise ValueError('auto_normalize must be bool valuable.')
        self._auto_normalize = value
        self._refresh_display()

    @property
    def display_update_time(self):
        """
        Time in seconds between preview display updates.

        :rtype: float
        """
        return self._display_update_time

    @display_update_time.setter
    def display_update_time(self, value):
        if value <= 0:
            raise ValueError('Display update time must be greater than zero seconds.')
        self._display_update_time = value

    cpdef object initialise(self, tuple pixels, int pixel_samples, double min_wavelength, double max_wavelength, int spectral_bins, list spectral_slices, bint quiet):

        nx, ny = pixels
        self._pixels = pixels
        self._samples = pixel_samples

        # create intermediate and final frame-buffers
        if not self.accumulate or self.rgb_frame is None or self.rgb_frame.shape != (nx, ny, 3):
            self.rgb_frame = StatsArray3D(nx, ny, 3)

        self._working_mean = np.zeros((nx, ny, 3))
        self._working_variance = np.zeros((nx, ny, 3))
        self._working_touched = np.zeros((nx, ny), dtype=np.int8)

        # generate pixel processor configurations for each spectral slice
        resampled_rgbs = [resample_phantom_rgb(slice.min_wavelength, slice.max_wavelength, slice.bins) for slice in spectral_slices]
        self._processors = [RGBPixelProcessor(resampled_rgb, self._exposure_time) for resampled_rgb in resampled_rgbs]

        self._quiet = quiet

        if self.display_progress:
            self._start_display()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef PixelProcessor pixel_processor(self, int x, int y, int slice_id):
        cdef RGBPixelProcessor processor = self._processors[slice_id]
        processor.reset()
        return processor

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef object update(self, int x, int y, int slice_id, tuple packed_result):

        cdef:
            double[::1] mean, variance

        # unpack results
        mean, variance = packed_result

        # accumulate sub-samples
        self._working_mean[x, y, 0] += mean[0]
        self._working_mean[x, y, 1] += mean[1]
        self._working_mean[x, y, 2] += mean[2]

        self._working_variance[x, y, 0] += variance[0]
        self._working_variance[x, y, 1] += variance[1]
        self._working_variance[x, y, 2] += variance[2]

        # mark pixel as modified
        self._working_touched[x, y] = 1

        # update users
        if self.display_progress:
            self._update_display(x, y)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef object finalise(self):

        cdef int x, y

        # update final frame with working frame results
        for x in range(self.rgb_frame.nx):
            for y in range(self.rgb_frame.ny):
                if self._working_touched[x, y] == 1:
                    self.rgb_frame.combine_samples(x, y, 0, self._working_mean[x, y, 0], self._working_variance[x, y, 0], self._samples)
                    self.rgb_frame.combine_samples(x, y, 1, self._working_mean[x, y, 1], self._working_variance[x, y, 1], self._samples)
                    self.rgb_frame.combine_samples(x, y, 2, self._working_mean[x, y, 2], self._working_variance[x, y, 2], self._samples)

        if self.display_progress:
            self._render_display(self.rgb_frame)

    cpdef object _start_display(self):
        """
        Display live render.
        """

        # reset figure handle if we are not persisting across observation runs
        if not self.display_persist_figure:
            self._display_figure = None

        # populate live frame with current frame state
        self._display_frame = self.rgb_frame.copy()

        # display initial frame
        self._render_display(self._display_frame, 'rendering...')

        # workaround for interactivity for QT backend
        try:
            plt.pause(0.1)
        except NotImplementedError:
            pass

        self._display_timer = time()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef object _update_display(self, int x, int y):
        """
        Update live render.
        """

        # update display pixel by combining existing frame data with working data
        self._display_frame.mean_mv[x, y, :] = self.rgb_frame.mean_mv[x, y, :]
        self._display_frame.variance_mv[x, y, :] = self.rgb_frame.variance_mv[x, y, :]
        self._display_frame.samples_mv[x, y, :] = self.rgb_frame.samples_mv[x, y, :]

        self._display_frame.combine_samples(x, y, 0, self._working_mean[x, y, 0], self._working_variance[x, y, 0], self._samples)
        self._display_frame.combine_samples(x, y, 1, self._working_mean[x, y, 1], self._working_variance[x, y, 1], self._samples)
        self._display_frame.combine_samples(x, y, 2, self._working_mean[x, y, 2], self._working_variance[x, y, 2], self._samples)

        # update live render display
        if (time() - self._display_timer) > self.display_update_time:

            if not self._quiet:
                print("{} - updating display...".format(self.name))

            self._render_display(self._display_frame, 'rendering...')

            # workaround for interactivity for QT backend
            try:
                plt.pause(0.1)
            except NotImplementedError:
                pass

            self._display_timer = time()

    cpdef object _refresh_display(self):
        """
        Refreshes the display window (if active) and frame data is present.

        This method is called when display attributes are changed to refresh
        the display according to the new settings.
        """

        # there must be frame data present
        if not self.rgb_frame:
            return

        # is there a figure present (only present if display() called or display progress was on during render)?
        if not self._display_figure:
            return

        # does the figure have an active window?
        if not plt.fignum_exists(self._display_figure.number):
            return

        self._render_display(self.rgb_frame)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object _render_display(self, StatsArray3D frame, str status=None):

        cdef np.ndarray[DTYPE_t, ndim=3] image

        INTERPOLATION = 'nearest'

        # generate display image
        image = self._generate_display_image(frame)

        # create a fresh figure if the existing figure window has gone missing
        if not self._display_figure or not plt.fignum_exists(self._display_figure.number):
            self._display_figure = plt.figure(facecolor=(0.5, 0.5, 0.5), figsize=_DISPLAY_SIZE, dpi=_DISPLAY_DPI)
        fig = self._display_figure

        # set window title
        if status:
            fig.canvas.set_window_title("{} - {}".format(self.name, status))
        else:
            fig.canvas.set_window_title(self.name)

        # populate figure
        fig.clf()
        ax = fig.add_axes([0,0,1,1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(np.transpose(image, (1, 0, 2)), aspect="equal", origin="upper", interpolation=INTERPOLATION)
        fig.canvas.draw_idle()
        plt.show()

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef np.ndarray[DTYPE_t, ndim=3] _generate_display_image(self, StatsArray3D frame):

        cdef:
            int x, y, c, nx, ny, nz
            np.ndarray[DTYPE_t, ndim=3] rgb_image, srgb_image
            double[:,:,::1] rgb_image_mv
            double peak_value

        rgb_image = frame.mean.copy()
        rgb_image_mv = rgb_image
        nx = rgb_image_mv.shape[0]
        ny = rgb_image_mv.shape[1]
        nz = rgb_image_mv.shape[2]

        # normalized if auto_normalize is True
        if self._auto_normalize is True:
            # calculate peak pixel value
            peak_value = self._calculate_maximum_pixel_value(rgb_image_mv)
            if peak_value > 0.0:
                for x in range(nx):
                    for y in range(ny):
                        for c in range(nz):
                            rgb_image_mv[x, y, c] *= 4095 / peak_value

        # convert Phantom RGB to sRGB
        srgb_image = self._generate_srgb_image(rgb_image_mv)

        return srgb_image

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef double _calculate_maximum_pixel_value(self, double[:, :, ::1] image_mv):

        cdef:
            int nx, ny, nz, x, y, c
            double peak_value = 0.0

        nx = image_mv.shape[0]
        ny = image_mv.shape[1]
        nz = image_mv.shape[2]

        for x in range(nx):
            for y in range(ny):
                for c in range(nz):
                    if image_mv[x, y, c] > peak_value:
                        peak_value = image_mv[x, y, c]

        return peak_value

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef np.ndarray[DTYPE_t, ndim=3] _generate_srgb_image(self, double[:,:,::1] image_mv):

        cdef:
            int nx, ny, ix, iy
            np.ndarray[DTYPE_t, ndim=3] srgb_image
            double[:,:,::1] srgb_image_mv
            (double, double, double) rgb_pixel

        nx = image_mv.shape[0]
        ny = image_mv.shape[1]
        srgb_image = np.zeros((nx, ny, 3))
        srgb_image_mv = srgb_image

        # convert to sRGB colour space
        for ix in range(nx):
            for iy in range(ny):

                rgb_pixel = phantom_rgb_to_srgb(
                    image_mv[ix, iy, 0],
                    image_mv[ix, iy, 1],
                    image_mv[ix, iy, 2]
                )

                srgb_image_mv[ix, iy, 0] = rgb_pixel[0]
                srgb_image_mv[ix, iy, 1] = rgb_pixel[1]
                srgb_image_mv[ix, iy, 2] = rgb_pixel[2]

        return srgb_image

    cpdef object display(self):
        """
        Plot the RGB frame.
        """
        if not self.rgb_frame:
            raise ValueError("There is no frame to display.")
        self._render_display(self.rgb_frame)

    cpdef object save(self, str filename):
        """
        Saves the display image to a png file.

        The current display settings (exposure, gamma, etc..) are used to
        process the image prior saving.

        :param str filename: Image path and filename.
        """
        cdef:
            np.ndarray[DTYPE_t, ndim=3] image

        if not self.rgb_frame:
            raise ValueError("There is no frame to save.")

        image = self._generate_display_image(self.rgb_frame)
        plt.imsave(filename, np.transpose(image, (1, 0, 2)))


cdef class RGBPixelProcessor(PixelProcessor):
    """
    PixelProcessor that converts each pixel's spectrum into three
    Phantom Camera's RGB colourspace values (12bit).
    """

    def __init__(self, double[:,::1] resampled_rgb not None, double exposure_time):
        self.resampled_rgb = resampled_rgb
        self.exposure_time = exposure_time
        self.rgb = StatsArray1D(3)

    cpdef object reset(self):
        self.rgb.clear()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef object add_sample(self, Spectrum spectrum, double sensitivity):

        cdef:
            int index
            double r, g, b

        r = 0.0
        g = 0.0
        b = 0.0

        # convert spectrum to Phantom RGB and add sample to pixel buffer
        for index in range(spectrum.bins):
            r += spectrum.delta_wavelength * spectrum.samples_mv[index] * self.resampled_rgb[index, 0]
            g += spectrum.delta_wavelength * spectrum.samples_mv[index] * self.resampled_rgb[index, 1]
            b += spectrum.delta_wavelength * spectrum.samples_mv[index] * self.resampled_rgb[index, 2]
        self.rgb.add_sample(0, r * 2.0 * M_PI * self.exposure_time / 6.15e-9)
        self.rgb.add_sample(1, g * 2.0 * M_PI * self.exposure_time / 6.15e-9)
        self.rgb.add_sample(2, b * 2.0 * M_PI * self.exposure_time / 6.15e-9)

    cpdef tuple pack_results(self):
        return self.rgb.mean, self.rgb.variance
