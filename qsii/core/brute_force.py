"""
brute_force.py

enumerate all 2^m elements of the stabilizer group
and count by Pauli weight. Used to verify transfer matrix results.

Scales as O(2^m) where m = number of generators. Use for n <= ~16.
"""
import numpy as np
from itertools import product
from .polynomial import Polynomial


def weight_enumerator_brute_force(generators, n: int) -> Polynomial:
    """
    Compute A(z) by iterating over all 2^m stabilizer group elements.

    Args:
        generators: list of (x_vec, z_vec) pairs, each length-n numpy array
        n: number of qubits

    Returns:
        A(z) = sum_j A_j z^j
    """
    m = len(generators)
    counts = [0] * (n + 1)

    for bits in product([0, 1], repeat=m):
        x = np.zeros(n, dtype=int)
        z = np.zeros(n, dtype=int)
        for b, (gx, gz) in zip(bits, generators):
            if b:
                x = (x + gx) % 2
                z = (z + gz) % 2
        weight = int(np.sum(((x | z) > 0).astype(int)))
        counts[weight] += 1

    return Polynomial(counts)


def complete_weight_enumerator_brute_force(generators, n: int) -> dict:
    """
    Compute D(W,X,Y,Z): track each Pauli type (I,X,Y,Z) separately.

    Returns:
        dict mapping (nI, nX, nY, nZ) -> count
    """
    m = len(generators)
    result = {}

    for bits in product([0, 1], repeat=m):
        x = np.zeros(n, dtype=int)
        z = np.zeros(n, dtype=int)
        for b, (gx, gz) in zip(bits, generators):
            if b:
                x = (x + gx) % 2
                z = (z + gz) % 2
        nX = int(np.sum((x == 1) & (z == 0)))
        nZ = int(np.sum((x == 0) & (z == 1)))
        nY = int(np.sum((x == 1) & (z == 1)))
        nI = n - nX - nZ - nY
        key = (nI, nX, nY, nZ)
        result[key] = result.get(key, 0) + 1

    return result


def double_enumerator_brute_force(generators, n: int, sector: str = 'Z') -> Polynomial:
    """
    Compute the CSS sector enumerator D_Z(z) or D_X(z).
    For CSS codes: tracks only the Z- or X-component weight.
    """
    m = len(generators)
    counts = [0] * (n + 1)

    for bits in product([0, 1], repeat=m):
        x = np.zeros(n, dtype=int)
        z = np.zeros(n, dtype=int)
        for b, (gx, gz) in zip(bits, generators):
            if b:
                x = (x + gx) % 2
                z = (z + gz) % 2
        w = int(np.sum(z)) if sector == 'Z' else int(np.sum(x))
        counts[w] += 1

    return Polynomial(counts)
