"""Subpackage for Inversion Problem."""
from .gcv import GCV
from .inversion import _SVDBase
from .lcurve import Lcurve

__all__ = ["_SVDBase", "Lcurve", "GCV"]
