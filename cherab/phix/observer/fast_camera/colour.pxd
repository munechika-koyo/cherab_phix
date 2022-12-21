from raysect.optical.spectrum cimport Spectrum


cpdef double[:,::1] resample_phantom_rgb(double min_wavelength, double max_wavelength, int bins)

cpdef (double, double, double) spectrum_to_phantom_rgb(
    Spectrum spectrum,
    double[:,::1] resampled_rgb=*,
    double exposure_time=*,
    double pixel_area=*
)

cpdef (double, double, double) phantom_rgb_to_srgb(double r, double g, double b)
