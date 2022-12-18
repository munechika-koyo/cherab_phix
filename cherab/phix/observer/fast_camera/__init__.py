"""Subpackage for Fast Camera."""
from .camera import import_phix_camera
from .colour import (
    phantom_rgb_to_srgb,
    plot_RGB_filter,
    plot_samples,
    resample_phantom_rgb,
    spectrum_to_phantom_rgb,
)
from .pipeline.rgb import RGBPipeline2D

__all__ = [
    "import_phix_camera",
    "resample_phantom_rgb",
    "spectrum_to_phantom_rgb",
    "phantom_rgb_to_srgb",
    "RGBPipeline2D",
    "plot_samples",
    "plot_RGB_filter",
]
