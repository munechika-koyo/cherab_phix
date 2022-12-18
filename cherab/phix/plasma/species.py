"""Module to offer the class taking species for PHiX."""
from __future__ import annotations

import numpy as np
from cherab.core import Maxwellian, Species, elements
from cherab.core.math import Constant3D, ConstantVector3D, Interpolate1DLinear, sample3d
from cherab.tools.equilibrium import EFITEquilibrium
from matplotlib import pyplot as plt
from raysect.core.math.function.float.function3d import Function3D
from raysect.core.math.function.vector3d.function3d import Function3D as VectorFunction3D
from raysect.optical import Vector3D
from scipy.constants import Boltzmann, atomic_mass, convert_temperature, electron_mass

from cherab.phix.machine.wall_outline import INNER_LIMITER

__all__ = ["PHiXSpecies"]


class PHiXSpecies:
    """Class representing PHiX plasma species.

    Parameters
    ------------
    equilibrium
        EFIT equilibrium object, default None

    Attributes
    -----------
    electron_distribution : :obj:`~raysect.core.math.function.float.function3d.base.Function3D`
        electron distribution function
    composition : list of :obj:`~cherab.core.Species`
        composition of plasma species, each information of whixh is
        element, charge, density_distribution, temperature_distribution, bulk_velocity_distribution.
    """

    def __init__(self, equilibrium: EFITEquilibrium | None = None):
        if equilibrium is None:
            raise ValueError("equilibrium argument must be input.")

        self.eq = equilibrium
        self.te_range = (42, 16)
        self.ne_range = (4.6e18, 3.5e17)

        psi = np.linspace(0, 1)

        # mapping electron temperture and density on flux surface
        self.temp1d = Interpolate1DLinear(
            psi, (self.te_range[1] - self.te_range[0]) * psi**2 + self.te_range[0]
        )
        self.dens1d = Interpolate1DLinear(
            psi, (self.ne_range[1] - self.ne_range[0]) * psi**2 + self.ne_range[0]
        )
        temperature = self.eq.map3d(self.temp1d, value_outside_lcfs=self.te_range[1])
        e_density = self.eq.map3d(self.dens1d)

        # assuming neutral hydrogen's density is equal to electron density
        h1_density = e_density

        # assuming hydrogen atomic density is derived from background gass pressure & lab temperature
        _background_pressure = 1.0e-2  # nitrogen pressure [Pa]
        _lab_Temp = 25  # [Celsius]
        h0_density = Constant3D(
            _background_pressure
            / 0.49  # nitrogen pressure -> hydrogen pressure
            / (Boltzmann * convert_temperature(_lab_Temp, "Celsius", "Kelvin"))
        )

        bulk_velocity = ConstantVector3D(Vector3D(0, 0, 0))

        # set electron distribution assuming Maxwellian
        self.electron_distribution = Maxwellian(
            e_density, temperature, bulk_velocity, electron_mass
        )

        # initialize composition
        self.composition = []
        # append species to composition list
        # H
        self.set_species(
            element="hydrogen",
            charge=0,
            density=h0_density,
            temperature=temperature,
            bulk_velocity=bulk_velocity,
        )
        # H+
        self.set_species(
            element="hydrogen",
            charge=1,
            density=h1_density,
            temperature=temperature,
            bulk_velocity=bulk_velocity,
        )
        # C iii (C+2)
        # self.set_species(
        #     element="carbon",
        #     charge=2,
        #     density=h0_density,
        #     temperature=temperature,
        #     bulk_velocity=bulk_velocity,
        # )

    def __repr__(self):
        return f"{self.composition}"

    def set_species(
        self,
        element: str | None = None,
        charge: int = 0,
        density: Function3D = Constant3D(1.0e19),
        temperature: Function3D = Constant3D(1.0e2),
        bulk_velocity: VectorFunction3D = ConstantVector3D(Vector3D(0, 0, 0)),
    ):
        """add species to composition which is assumed to be Maxwellian
        distribution.

        Parameters
        ------------
        element
            element name registored in cherabs elements.pyx, by default None
        charge
            element's charge state, by default 0
        density
            density distribution, by default Constant3D(1.0e19)
        temperature
            temperature distribution, by default Constant3D(1.0e2)
        bulk_velocity
            bulk velocity, by default ConstantVector3D(0)
        """

        if element is None:
            message = "Parameter 'element' is required to be input."
            raise ValueError(message)

        try:
            # extract specified element object
            element = getattr(elements, element)
        except AttributeError:
            message = (
                f"element name '{element}' is not implemented."
                f"You can implement manually using Element class"
            )
            raise NotImplementedError(message)

        # element mass
        element_mass = element.atomic_weight * atomic_mass

        # Maxwellian distribution
        distribution = Maxwellian(density, temperature, bulk_velocity, element_mass)

        # append plasma.composition
        self.composition.append(Species(element, charge, distribution))

    def plot_distribution(self, res: float = 0.001):
        """plot species temperature & density profile.

        Parameters
        ----------
        res
            Spactial resolution for sampling, by default 0.001 [m]
        """
        # grid
        r_min, r_max = self.eq.r_range
        z_min, z_max = self.eq.z_range
        nr = int((r_max - r_min) / res)
        nz = int((z_max - z_min) / res)

        # electron sampling
        r, _, z, dens = sample3d(
            self.electron_distribution.density, (r_min, r_max, nr), (0, 0, 1), (z_min, z_max, nz)
        )
        r, _, z, temp = sample3d(
            self.electron_distribution.effective_temperature,
            (r_min, r_max, nr),
            (0, 0, 1),
            (z_min, z_max, nz),
        )

        # create inner limiter mask array
        mask = np.zeros_like(dens, dtype=bool)
        for ir, r_tmp in enumerate(r):
            for iz, z_tmp in enumerate(z):
                mask[ir, 0, iz] = not self.eq.inside_limiter(r_tmp, z_tmp)

        # plot
        for sample, title, clabel in zip(
            [dens, temp],
            ["electron density[1/m$^3$]", "electron temperature[eV]"],
            ["density [1/m$^3$]", "temperature [eV]"],
        ):
            plt.figure()
            plt.axis("equal")
            plt.pcolormesh(r, z, np.squeeze(np.ma.masked_array(sample, mask)).T, shading="gouraud")
            plt.autoscale(tight=True)
            plt.colorbar(label=clabel)
            plt.plot(INNER_LIMITER[:, 0], INNER_LIMITER[:, 1])
            plt.title(title)
            plt.xlabel("R (meters)")
            plt.ylabel("Z (meters)")

        # species sampling
        for species in self.composition:
            r, _, z, dens = sample3d(
                species.distribution.density, (r_min, r_max, nr), (0, 0, 1), (z_min, z_max, nz)
            )
            r, _, z, temp = sample3d(
                species.distribution.effective_temperature,
                (r_min, r_max, nr),
                (0, 0, 1),
                (z_min, z_max, nz),
            )

            # plot
            for sample, title, clabel in zip(
                [dens, temp],
                [
                    f"{species.element.name}+{species.charge} density [1/m$^3$]",
                    f"{species.element.name}+{species.charge} temperature [eV]",
                ],
                ["density [1/m$^3$]", "temperature [eV]"],
            ):
                plt.figure()
                plt.axis("equal")
                plt.pcolormesh(
                    r, z, np.squeeze(np.ma.masked_array(sample, mask)).T, shading="gouraud"
                )
                plt.autoscale(tight=True)
                plt.colorbar(label=clabel)
                plt.plot(INNER_LIMITER[:, 0], INNER_LIMITER[:, 1])
                plt.title(title)
                plt.xlabel("R (meters)")
                plt.ylabel("Z (meters)")

        plt.show()

    def plot_1d_profile(self):
        """plot r vs electron density or temperature 1D profile."""
        eq = self.eq
        funcs = [
            self.electron_distribution.density,
            self.electron_distribution.effective_temperature,
        ]
        for func, ylabel, title in zip(
            funcs,
            ["density [1/m$^3$]", "temperature [eV]"],
            ["electron density", "electron temperature"],
        ):
            r, _, _, sample = sample3d(
                func,
                (eq.magnetic_axis.x, eq.limiter_polygon[:, 0].max(), 100),
                (0, 0, 1),
                (eq.magnetic_axis.y, eq.magnetic_axis.y, 1),
            )
            plt.figure()
            plt.plot(r, np.squeeze(sample))
            plt.xlabel("R [m]")
            plt.ylabel(ylabel)
            plt.title(title)

        plt.show()


# For debugging
if __name__ == "__main__":
    from cherab.phix.plasma import import_equilibrium

    eq = import_equilibrium()
    species = PHiXSpecies(eq)
    species.plot_distribution()
