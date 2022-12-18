"""Module defining metal material classes."""
import json
from pathlib import Path

from numpy import array
from raysect.optical import InterpolatedSF
from raysect.optical.material import Conductor

__all__ = ["SUS316L"]


class _DataLoader(Conductor):
    def __init__(self, filename):
        with open(Path(__file__).parent.resolve() / "data" / f"{filename}.json", "r") as f:
            data = json.load(f)

        wavelength = array(data["wavelength"])
        index = InterpolatedSF(wavelength, array(data["index"]))
        extinction = InterpolatedSF(wavelength, array(data["extinction"]))

        super().__init__(index, extinction)


class SUS316L(_DataLoader):
    """Stainless Used Steel 316L metal material."""

    def __init__(self):
        super().__init__("sus316L")
