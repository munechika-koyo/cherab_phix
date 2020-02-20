from scipy.constants import electron_mass, atomic_mass

from matplotlib import pyplot as plt
import numpy as np

# Cherab and raysect imports
from cherab.core import Species, Maxwellian, elements
from cherab.core.math import Constant3D, ConstantVector3D
from cherab.core.math import Interpolate1DLinear
from cherab.core.math import sample3d
from raysect.optical import Vector3D

from cherab.phix.machine.wall_outline import INNER_LIMITER


class PHiXSpecies:
    """Class representing PHiX plasma species

    Parameters
    ------------
    equilibrium : object, required
        EFIT equilibrium object, default None

    Attributes
    -----------
    electron_distribution : Function3D
        electron distribution function
    composition : list
        composition of plasma species
    """

    def __init__(self, equilibrium=None):
        self.eq = equilibrium
        # mapping temp. and density on flux surface
        temp1d = Interpolate1DLinear([0, 1], [42, 16])
        dens1d = Interpolate1DLinear([0, 1], [4.6e18, 3.5e17])
        temperature = self.eq.map3d(temp1d, value_outside_lcfs=16)
        e_density = self.eq.map3d(dens1d)
        h1_density = e_density
        h0_density = Constant3D(3e17)

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
        self.set_species(
            element="carbon",
            charge=2,
            density=h0_density,
            temperature=temperature,
            bulk_velocity=bulk_velocity,
        )

    def __repr__(self):
        return f"{self.composition}"

    def set_species(
        self,
        element=None,
        charge=0,
        density=Constant3D(1.0e19),
        temperature=Constant3D(1.0e2),
        bulk_velocity=ConstantVector3D(Vector3D(0, 0, 0)),
    ):
        """add species to composition which is assumed to be Maxwellian distribution.

        Parameters
        ------------
        element : str, required
            element name registored in cherabs elements.pyx, by default None
        charge : int, required
            element's charge state, by default 0
        density : Function3D, optional
            density distribution, by default Constant3D(1.0e19)
        temperature : Function3D, optional
            temperature distribution, by default Constant3D(1.0e2)
        bulk_velocity : VectorFunction3D, optional
            bulk velocity, by default ConstantVector3D(0)
        """

        if element is None:
            message = f"Parameter 'element' is required to be input."
            raise ValueError(message)

        try:
            # extract specified element object
            element = eval(f"elements.{element}")
        except Exception:
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

    def plot_distribution(self, res=0.001):
        """plot species temperature & density profile

        Parameters
        ----------
        res : float, optional
            Spactial resolution for sampling, by default 0.001
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

        # plot
        for sample, title in zip(
            [dens, temp], ["electron density[1/m$^3$]", "electron temperature[eV]"]
        ):
            plt.figure()
            plt.axis("equal")
            plt.pcolormesh(r, z, np.squeeze(sample).T, shading="gouraud")
            plt.autoscale(tight=True)
            plt.colorbar()
            plt.plot(INNER_LIMITER[:, 0], INNER_LIMITER[:, 1])
            plt.title(title)
            plt.xlabel("R (meters)")
            plt.xlabel("Z (meters)")

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
            for sample, title in zip(
                [dens, temp],
                [
                    f"{species.element.name}+{species.charge} density [1/m$^3$]",
                    f"{species.element.name}+{species.charge} temperature [eV]",
                ],
            ):
                plt.figure()
                plt.axis("equal")
                plt.pcolormesh(r, z, np.squeeze(sample).T, shading="gouraud")
                plt.autoscale(tight=True)
                plt.colorbar()
                plt.plot(INNER_LIMITER[:, 0], INNER_LIMITER[:, 1])
                plt.title(title)
                plt.xlabel("R (meters)")
                plt.xlabel("Z (meters)")

    def plot_1d_profile(self):
        """plot r vs electron density or temperature 1D profile
        """
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
