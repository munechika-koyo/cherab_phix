"""Module defining Rough metal material classes."""
import json
from pathlib import Path

from numpy import array
from raysect.optical import InterpolatedSF
from raysect.optical.material import RoughConductor

__all__ = ["RoughSUS316L"]


class _DataLoader(RoughConductor):
    def __init__(self, filename, roughness):
        with open(Path(__file__).parent.resolve() / "data" / f"{filename}.json", "r") as f:
            data = json.load(f)

        wavelength = array(data["wavelength"])
        index = InterpolatedSF(wavelength, array(data["index"]))
        extinction = InterpolatedSF(wavelength, array(data["extinction"]))

        super().__init__(index, extinction, roughness)


class RoughSUS316L(_DataLoader):
    """Stainless Used Steel 316L metal material."""

    def __init__(self, roughness):
        super().__init__("sus316L", roughness)
