"""
macwilliams.py

MacWilliams identity for quantum stabilizer codes (Shor & Laflamme, PRL 1997).

For a stabilizer code with enumerator A(z):
    B(z) = (1 / 2^(n-k)) * (1+z)^n * A((1-z)/(1+z))

Implemented via Krawtchouk polynomials for numerical stability.
"""
from math import comb
import numpy as np
from .polynomial import Polynomial


def macwilliams_transform(A: Polynomial, n: int, k: int = 0) -> Polynomial:
    """
    Compute the dual enumerator B(z) from A(z).

    Args:
        A: primal enumerator
        n: number of qubits
        k: number of logical qubits (0 for pure stabilizer states)

    Returns:
        B: dual enumerator
    """
    K = _krawtchouk_matrix(n)
    a_vec = np.zeros(n + 1)
    for i, c in enumerate(A.coeffs):
        if i <= n:
            a_vec[i] = c
    b_vec = (1.0 / 2 ** (n - k)) * (K @ a_vec)
    return Polynomial(b_vec.tolist())


def _krawtchouk_matrix(n):
    K = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            K[i, j] = _krawtchouk(j, i, n)
    return K


def _krawtchouk(k, x, n):
    total = 0.0
    for s in range(k + 1):
        if s > x or (k - s) > (n - x):
            continue
        total += ((-1) ** s) * comb(x, s) * comb(n - x, k - s)
    return total
