"""Module defining dielectric material classes."""
import json
from pathlib import Path

from numpy import array
from raysect.optical import InterpolatedSF
from raysect.optical.material import Dielectric

__all__ = ["PCTFE"]


class _DataLoader(Dielectric):
    def __init__(self, filename):
        with open(Path(__file__).parent.resolve() / "data" / f"{filename}.json", "r") as f:
            data = json.load(f)

        wavelength = array(data["wavelength"])
        index = InterpolatedSF(wavelength, array(data["index"]))
        transmittance = InterpolatedSF(wavelength, array(data["transmittance"]))
        super().__init__(index, transmittance)


class PCTFE(_DataLoader):
    """Polychlorotrifluoroethylene material."""

    def __init__(self):
        super().__init__("PCTFE")
