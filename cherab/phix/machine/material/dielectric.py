from os import path
import json
from numpy import array
from raysect.optical import InterpolatedSF
from raysect.optical.material import Dielectric


class _DataLoader(Dielectric):
    def __init__(self, filename):

        with open(path.join(path.dirname(__file__), "data", filename + ".json")) as f:
            data = json.load(f)

        wavelength = array(data["wavelength"])
        index = InterpolatedSF(wavelength, array(data["index"]))
        transmittance = InterpolatedSF(wavelength, array(data["transmittance"]))
        super().__init__(index, transmittance)


class PCTFE(_DataLoader):
    """Polychlorotrifluoroethylene material."""

    def __init__(self):
        super().__init__("PCTFE")
