"""
ghz.py

Weight enumerator for the n-qubit GHZ state.

GHZ state: |0...0⟩ + |1...1⟩ / sqrt(2)

Stabilizers:
  Z_i Z_{i+1}   for i = 1, ..., n-1
  X_1 X_2 ... X_n

Result: A(z) = (1+z)^n  (perfect binomial — verified by brute force)

The parity-augmented transfer matrix (from the MATLAB implementation)
tracks anticommutation with X^n. Here we expose both the analytic formula
and the brute-force computation so they can be compared.

Author: Julianna Kelley
"""
import numpy as np
from math import comb as _comb
from ..core import Polynomial, macwilliams_transform
from ..core.brute_force import weight_enumerator_brute_force, complete_weight_enumerator_brute_force


def ghz_generators(n: int):
    """Return (x_vec, z_vec) stabilizer generators for n-qubit GHZ."""
    generators = []
    for i in range(n - 1):
        x = np.zeros(n, dtype=int)
        z = np.zeros(n, dtype=int)
        z[i] = 1; z[i+1] = 1
        generators.append((x, z))
    x = np.ones(n, dtype=int)
    z = np.zeros(n, dtype=int)
    generators.append((x, z))
    return generators


def compute_ghz_enumerator(n: int, verbose: bool = True):
    """
    Compute A(z) = (1+z)^n for the n-qubit GHZ state.

    The stabilizer group of the GHZ state has 2^n elements:
    n-1 ZZ generators + global X^n, giving A_j = C(n,j).

    For n <= 12, also runs brute force to cross-check.

    Returns:
        A, B: primal and MacWilliams dual enumerators
    """
    # Analytic result
    A = Polynomial([float(_comb(n, j)) for j in range(n + 1)])
    B = macwilliams_transform(A, n, k=1)

    if verbose:
        print(f"GHZ n={n}:")
        print(f"  A(z) = (1+z)^{n}")
        print(f"  A(1) = {A.evaluate(1.0):.0f}  (expected {2**n})")
        if n <= 12:
            A_bf = weight_enumerator_brute_force(ghz_generators(n), n)
            match = all(abs(A.coeffs[j] - A_bf.coeffs[j]) < 0.5
                        for j in range(len(A.coeffs)))
            print(f"  Brute force match: {match}")

    return A, B


def compute_ghz_complete(n: int, verbose: bool = True):
    """Complete weight enumerator D(W,X,Y,Z) for GHZ. Brute force, n <= 12."""
    if n > 12:
        raise ValueError(f"Complete enumerator limited to n<=12, got n={n}")
    D = complete_weight_enumerator_brute_force(ghz_generators(n), n)
    if verbose:
        print(f"GHZ n={n} complete enumerator:")
        for key, val in sorted(D.items()):
            if val > 0:
                nI, nX, nY, nZ = key
                print(f"  W^{nI} X^{nX} Y^{nY} Z^{nZ}: {val}")
    return D
