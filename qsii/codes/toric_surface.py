"""
toric_surface.py

Weight enumerators for toric and surface codes, with statistical mechanics verification.

Toric code: [[2L^2, 2, L]]  CSS code on L x L torus
Surface code: CSS code on L x L open lattice with boundaries

Author: Julianna Kelley
"""
import numpy as np
from itertools import product as iproduct
from ..core import Polynomial, macwilliams_transform
from ..core.brute_force import (
    weight_enumerator_brute_force,
    complete_weight_enumerator_brute_force,
    double_enumerator_brute_force,
)

def toric_code_generators(L: int):
    """
    Stabilizer generators for the toric code on an L x L torus.
    n = 2L^2 qubits (L^2 horizontal + L^2 vertical edges).

    Z-plaquette stabilizers: Z on 4 edges around each face.
    X-star stabilizers: X on 4 edges around each vertex.
    """
    n = 2 * L * L

    def h(i, j):  # horizontal edge index
        return i * L + j

    def v(i, j):  # vertical edge index
        return L * L + i * L + j

    generators = []

    # Z plaquettes
    for i in range(L):
        for j in range(L):
            z_vec = np.zeros(n, dtype=int)
            z_vec[h(i, j)] = 1
            z_vec[h((i+1) % L, j)] = 1
            z_vec[v(i, j)] = 1
            z_vec[v(i, (j+1) % L)] = 1
            generators.append((np.zeros(n, dtype=int), z_vec))

    # X stars
    for i in range(L):
        for j in range(L):
            x_vec = np.zeros(n, dtype=int)
            x_vec[h(i, j)] = 1
            x_vec[h(i, (j-1) % L)] = 1
            x_vec[v(i, j)] = 1
            x_vec[v((i-1) % L, j)] = 1
            generators.append((x_vec, np.zeros(n, dtype=int)))

    return generators


def compute_toric_enumerators(L: int, verbose: bool = True):
    """
    Compute SL scalar, double (Z and X sector), and MacWilliams dual for toric code.

    Returns:
        A_SL, B_SL, D_Z, D_X
    """
    n = 2 * L * L
    generators = toric_code_generators(L)

    A_SL = weight_enumerator_brute_force(generators, n)
    B_SL = macwilliams_transform(A_SL, n, k=2)  # k=2: two logical qubits
    D_Z  = double_enumerator_brute_force(generators, n, sector='Z')
    D_X  = double_enumerator_brute_force(generators, n, sector='X')

    if verbose:
        print(f"Toric Code L={L} (n={n}, k=2):")
        print(f"  A_SL(z) = {A_SL}")
        print(f"  A_SL(1) = {A_SL.evaluate(1.0):.0f}  (expected {2**n})")
        print(f"  D_Z(z)  = {D_Z}")
        print(f"  D_X(z)  = {D_X}")
        print(f"  D_Z == D_X: {all(abs(D_Z.coeffs[i]-D_X.coeffs[i])<0.5 for i in range(min(len(D_Z.coeffs),len(D_X.coeffs))))}")

    return A_SL, B_SL, D_Z, D_X


def surface_code_generators(L: int):
    """
    Stabilizer generators for surface code on L x L open lattice.
    n = L*(L-1) + (L-1)*L = 2*L*(L-1) qubits.
    """
    rows, cols = L, L
    n_h = rows * (cols - 1)
    n_v = (rows - 1) * cols
    n = n_h + n_v

    def h(i, j):
        return i * (cols - 1) + j

    def v(i, j):
        return n_h + i * cols + j

    generators = []

    # Z plaquettes (interior faces)
    for i in range(rows - 1):
        for j in range(cols - 1):
            z_vec = np.zeros(n, dtype=int)
            z_vec[h(i, j)] = 1
            z_vec[h(i+1, j)] = 1
            z_vec[v(i, j)] = 1
            z_vec[v(i, j+1)] = 1
            generators.append((np.zeros(n, dtype=int), z_vec))

    # X stars (all vertices with >= 2 edges)
    for i in range(rows):
        for j in range(cols):
            edges = []
            if j < cols-1:  edges.append(h(i, j))
            if j > 0:       edges.append(h(i, j-1))
            if i < rows-1:  edges.append(v(i, j))
            if i > 0:       edges.append(v(i-1, j))
            if len(edges) < 2:
                continue
            x_vec = np.zeros(n, dtype=int)
            for e in edges:
                x_vec[e] = 1
            generators.append((x_vec, np.zeros(n, dtype=int)))

    return generators


def compute_surface_enumerators(L: int, verbose: bool = True):
    """Compute enumerators for surface code. Returns A_SL, B_SL, D_Z, D_X."""
    rows, cols = L, L
    n = rows * (cols - 1) + (rows - 1) * cols
    generators = surface_code_generators(L)

    A_SL = weight_enumerator_brute_force(generators, n)
    B_SL = macwilliams_transform(A_SL, n, k=1)
    D_Z  = double_enumerator_brute_force(generators, n, sector='Z')
    D_X  = double_enumerator_brute_force(generators, n, sector='X')

    if verbose:
        print(f"Surface Code L={L} (n={n}, k=1):")
        print(f"  A_SL(z) = {A_SL}")
        print(f"  D_Z(z)  = {D_Z}")
        print(f"  D_X(z)  = {D_X}")

    return A_SL, B_SL, D_Z, D_X


def ising_partition_function(L: int, beta: float, boundary: str = 'periodic') -> float:
    """
    2D Ising partition function Z = Σ_σ exp(β Σ_{<ij>} σ_i σ_j) on L x L lattice.
    """
    N = L * L
    Z = 0.0
    for spins_flat in iproduct([-1, 1], repeat=N):
        spins = np.array(spins_flat).reshape(L, L)
        E = 0.0
        for i in range(L):
            for j in range(L):
                if boundary == 'periodic' or j + 1 < L:
                    E += spins[i, j] * spins[i, (j+1) % L]
                if boundary == 'periodic' or i + 1 < L:
                    E += spins[i, j] * spins[(i+1) % L, j]
        Z += np.exp(beta * E)
    return Z


def verify_ising_ratio(L: int = 2):
    """
    Full stabilizer group D_Z (including X-stars) = 2^(L^2) * Z_Ising,
    because the 2^(L^2) X-star combinations all contribute z^0 to the Z-weight.
    This factor of 2^(L^2) reflects the X-sector degeneracy.
    """
    from ..core.brute_force import weight_enumerator_brute_force
    generators = toric_code_generators(L)
    n = 2 * L * L

    # Z-only generators (plaquettes)
    z_generators = generators[:L*L]
    D_Z = weight_enumerator_brute_force(z_generators, n)

    # Build Z_Ising as polynomial: z^k counts configs with k antiparallel bonds
    N = L * L
    from itertools import product as iproduct
    bond_counts = {}
    for spins in iproduct([-1, 1], repeat=N):
        s = np.array(spins).reshape(L, L)
        k = 0
        for i in range(L):
            for j in range(L):
                if s[i, j] != s[i, (j+1) % L]: k += 1
                if s[i, j] != s[(i+1) % L, j]: k += 1
        bond_counts[k] = bond_counts.get(k, 0) + 1

    print(f"\nToric Code L={L}: D_Z(z) vs Z_Ising(z) as polynomials")
    print(f"D_Z (Z-sector):  {D_Z}")
    zi_coeffs = [bond_counts.get(k, 0) for k in range(2*n + 1)]
    zi_poly_str = " + ".join(f"{c}z^{k}" for k, c in enumerate(zi_coeffs) if c > 0)
    print(f"Z_Ising (poly):  {zi_poly_str}")
    match = all(abs(D_Z.coeffs[k] - zi_coeffs[k]) < 0.5 for k in range(len(D_Z.coeffs)))
    print(f"Match: {match}  (D_Z = Z_Ising exactly)")
