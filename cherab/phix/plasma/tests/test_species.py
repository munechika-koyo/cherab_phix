import matplotlib
import pytest

matplotlib.use("Agg")

from cherab.phix.plasma.equilibrium import import_equilibrium
from cherab.phix.plasma.species import PHiXSpecies


@pytest.fixture
def species():
    equilibrium = import_equilibrium(model_variant="phix10")
    return PHiXSpecies(equilibrium)


class TestPHiXSpecies:
    @pytest.mark.parametrize(
        ["model_variant"],
        [
            pytest.param("phix10", id="phix10"),
            pytest.param("phix12", id="phix12"),
            pytest.param("phix13", id="phix13"),
            pytest.param("phix14", id="phix14"),
        ],
    )
    def test_init(self, model_variant):
        equilibrium = import_equilibrium(model_variant=model_variant)
        species = PHiXSpecies(equilibrium)
        assert len(species.composition) == 2

    @pytest.mark.parametrize(
        ["element", "charge"],
        [
            pytest.param("carbon", 2, id="C+2"),
            pytest.param("oxygen", 5, id="O+5"),
        ],
    )
    def test_set_species(self, species, element, charge):
        species.set_species(element=element, charge=charge)
        assert len(species.composition) == 3
        assert species.composition[2].element.name == element
        assert species.composition[2].charge == charge

    def test_plot_distribution(self, species):
        species.plot_distribution()

    def test_plot_1d_profile(self, species):
        species.plot_1d_profile()
