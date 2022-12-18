"""Subpackage for Material classes."""
from .dielectric import PCTFE
from .metal import SUS316L
from .roughmetal import RoughSUS316L

__all__ = ["RoughSUS316L", "SUS316L", "PCTFE"]
