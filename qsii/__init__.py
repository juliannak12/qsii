"""
qsii — Quantum Stabilizer Information and Invariants

Weight enumerators for quantum error-correcting codes
via tensor network contraction (Quantum LEGO framework).

"""
from .codes import (
    compute_ghz_enumerator,
    compute_cluster_1d_enumerator,
    compute_cluster_2d_enumerator,
    compute_toric_enumerators,
    compute_surface_enumerators,
    verify_ising_ratio,
)
from .core import Polynomial, macwilliams_transform

__version__ = "0.1.0"
__author__ = "Julianna Kelley"
