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

cimport numpy as np
from raysect.core.math cimport StatsArray1D, StatsArray3D
from raysect.optical.observer.base cimport Pipeline2D, PixelProcessor

ctypedef np.float64_t DTYPE_t

cdef class RGBPipeline2D(Pipeline2D):

    cdef:
        str name
        public bint display_progress
        double _display_timer
        double _display_update_time
        public bint accumulate
        readonly StatsArray3D rgb_frame
        double[:, :, ::1] _working_mean, _working_variance
        char[:, ::1] _working_touched
        StatsArray3D _display_frame
        list _processors
        tuple _pixels
        int _samples
        object _display_figure
        double _exposure_time
        bint _auto_normalize
        public bint display_persist_figure
        bint _quiet

    cpdef object _start_display(self)

    cpdef object _update_display(self, int x, int y)

    cpdef object _refresh_display(self)

    cpdef object _render_display(self, StatsArray3D frame, str status=*)

    cpdef np.ndarray[DTYPE_t, ndim=3] _generate_display_image(self, StatsArray3D frame)

    cpdef double _calculate_maximum_pixel_value(self, double[:, :, ::1] image_mv)

    cpdef np.ndarray[DTYPE_t, ndim=3] _generate_srgb_image(self, double[:, :, ::1] rgb_image_mv)

    cpdef object display(self)

    cpdef object save(self, str filename)


cdef class RGBPixelProcessor(PixelProcessor):

    cdef:
        double[:, ::1] resampled_rgb
        double exposure_time
        StatsArray1D rgb

    cpdef object reset(self)
