"""
cluster.py

Weight enumerators for 1D and 2D cluster states.

1D Cluster — transfer matrix tensor contraction

Stabilizers: K_i = X_i Z_{i-1} Z_{i+1}  (bulk); boundary terms drop one Z.

Bond dimension D=4 (one Pauli per bond). Transfer matrix is 16x16.
Entry T[ab, bc] = z^{wt(b)} if the triple (a, b, c) satisfies the
local stabilizer constraint for the bulk stabilizer Z X Z.

The constraint: a Pauli string commutes with Z_left X_center Z_right iff
the total number of anticommutations is even.

  Anticommutes with Z:  X (p=1) and Y (p=2)
  Anticommutes with X:  Y (p=2) and Z (p=3)

Contract as:   A(z) = v^T · T^(n-2) · f
where v fixes p_0 = I (left virtual) and f fixes p_{n+1} = I (right virtual).

Scales to 1000+ qubits via repeated squaring (O(log n) matrix multiplications).

2D Cluster — brute force with optimized stabilizer checking
Stabilizers: K_{i,j} = X_{i,j} Z_{i±1,j} Z_{i,j±1}
Exact for n = Lx*Ly <= ~16 qubits. Row-by-row contraction in progress.

Author: Julianna Kelley
"""
import numpy as np
from ..core import Polynomial, PolynomialMatrix, macwilliams_transform
from ..core.brute_force import weight_enumerator_brute_force, complete_weight_enumerator_brute_force

# Pauli weights: I=0, X=1, Y=1, Z=1
_WT = [0, 1, 1, 1]



def _cluster_local_ok(a, b, c):
    """
    Check if Pauli triple (a, b, c) commutes with the ZXZ stabilizer.

    Stabilizer: Z_{i-1} X_i Z_{i+1}
    A Pauli string commutes with ZXZ iff the number of anticommuting
    positions is even.

    Anticommutes with Z: X (1) or Y (2)
    Anticommutes with X: Y (2) or Z (3)
    """
    count = 0
    if a in (1, 2):   count += 1   # left Z position
    if b in (2, 3):   count += 1   # center X position
    if c in (1, 2):   count += 1   # right Z position
    return count % 2 == 0


def _build_transfer_matrix():
    """
    Build the 16x16 polynomial transfer matrix T for the 1D cluster state.

    State indices: ab = a*4 + b where a, b in {0,1,2,3} = {I,X,Y,Z}.
    T[ab, bc] = z^{wt(b)} if cluster_local_ok(a, b, c), else 0.
    """
    BD = 16
    T = PolynomialMatrix.zeros(BD)
    for a in range(4):
        for b in range(4):
            idx_ab = a * 4 + b
            for c in range(4):
                idx_bc = b * 4 + c
                if _cluster_local_ok(a, b, c):
                    coeffs = [0.0] * (_WT[b] + 1)
                    coeffs[_WT[b]] = 1.0
                    p = Polynomial(coeffs)
                    T.data[idx_ab][idx_bc] = T.data[idx_ab][idx_bc] + p
    return T


def compute_cluster_1d_enumerator(n: int, verbose: bool = True):
    """
    Compute A(z) for the n-qubit 1D cluster state via transfer matrix contraction.

    Directly translates the MATLAB clusterB function:
      1. Build 16x16 transfer matrix T
      2. Power T^(n-2) via repeated squaring
      3. Contract with boundary vectors: A(z) = v^T · T^(n-2) · f

    For n <= 12, also runs brute force as a cross-check.

    Returns:
        A, B: primal and dual (MacWilliams) enumerators
    """
    BD = 16

    T = _build_transfer_matrix()

    # Initial vector v: left boundary qubit is I (p_0 = I = 0)
    # v[ab] = 1 if a == 0 (left virtual index is I), for all b
    v = [Polynomial.zero() for _ in range(BD)]
    for b in range(4):
        v[0 * 4 + b] = Polynomial.one()

    # Final selector f: right boundary qubit is I (p_{n+1} = I = 0)
    # f[ab] = 1 if b == 0 (right virtual index is I), for all a
    f = [Polynomial.zero() for _ in range(BD)]
    for a in range(4):
        f[a * 4 + 0] = Polynomial.one()

    # Contract v^T · T^n · f  (MATLAB: T is multiplied n times)
    if n == 0:
        A = Polynomial.one()
    else:
        Tn = T.power(n)

        temp = [Polynomial.zero() for _ in range(BD)]
        for i in range(BD):
            if all(abs(c) < 1e-12 for c in v[i].coeffs):
                continue
            for j in range(BD):
                if all(abs(c) < 1e-12 for c in Tn.data[i][j].coeffs):
                    continue
                temp[j] = temp[j] + (v[i] * Tn.data[i][j])

        A = Polynomial.zero()
        for j in range(BD):
            if all(abs(c) < 1e-12 for c in temp[j].coeffs):
                continue
            A = A + (temp[j] * f[j])

    B = macwilliams_transform(A, n, k=0)

    if verbose:
        print(f"1D Cluster n={n}:")
        print(f"  A(z) = {A}")
        print(f"  A(1) = {A.evaluate(1.0):.0f}  (expected {2**n})")
        d = next((i for i, c in enumerate(A.coeffs) if i > 0 and abs(c) > 0.5), None)
        print(f"  min weight d = {d}")
        if n <= 12:
            # Cross-check with brute force
            gens = cluster_1d_generators(n)
            A_bf = weight_enumerator_brute_force(gens, n)
            match = all(abs(A.coeffs[i] - A_bf.coeffs[i]) < 0.5
                        for i in range(min(len(A.coeffs), len(A_bf.coeffs))))
            print(f"  Brute force match: {match}")

    return A, B


def cluster_1d_generators(n: int, periodic: bool = False):
    """Return (x_vec, z_vec) stabilizer generators for 1D cluster state."""
    generators = []
    for i in range(n):
        x = np.zeros(n, dtype=int)
        z = np.zeros(n, dtype=int)
        x[i] = 1
        if i > 0:
            z[i - 1] = 1
        elif periodic:
            z[n - 1] = 1
        if i < n - 1:
            z[i + 1] = 1
        elif periodic:
            z[0] = 1
        generators.append((x, z))
    return generators


def compute_cluster_1d_complete(n: int):
    """Complete weight enumerator D(W,X,Y,Z) for 1D cluster. Brute force, n <= 14."""
    if n > 14:
        raise ValueError(f"Complete enumerator limited to n<=14, got n={n}")
    return complete_weight_enumerator_brute_force(cluster_1d_generators(n), n)



def cluster_2d_generators(Lx: int, Ly: int):
    """
    Stabilizer generators for 2D cluster state on Lx x Ly grid (open boundary).
    K_{i,j} = X_{i,j} · Z_{neighbors}
    """
    n = Lx * Ly

    def idx(i, j):
        return i * Ly + j

    generators = []
    for i in range(Lx):
        for j in range(Ly):
            x_vec = np.zeros(n, dtype=int)
            z_vec = np.zeros(n, dtype=int)
            x_vec[idx(i, j)] = 1
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < Lx and 0 <= nj < Ly:
                    z_vec[idx(ni, nj)] = 1
            generators.append((x_vec, z_vec))
    return generators


def compute_cluster_2d_enumerator(Lx: int, Ly: int, verbose: bool = True):
    """
    Compute A(z) for the Lx x Ly 2D cluster state.
    Uses brute force stabilizer enumeration (exact, practical up to n~16).

    Returns:
        A, B: primal and dual enumerators
    """
    n = Lx * Ly
    if n > 18:
        raise ValueError(
            f"2D cluster brute force requires n <= 18, got n={n}={Lx}x{Ly}. "
            "Use a smaller grid or the planqTN server for larger systems."
        )

    generators = cluster_2d_generators(Lx, Ly)
    A = weight_enumerator_brute_force(generators, n)
    B = macwilliams_transform(A, n, k=0)

    if verbose:
        print(f"2D Cluster {Lx}x{Ly} (n={n}):")
        print(f"  A(z) = {A}")
        print(f"  A(1) = {A.evaluate(1.0):.0f}  (expected {2**n})")
        d = next((i for i, c in enumerate(A.coeffs) if i > 0 and abs(c) > 0.5), None)
        print(f"  min weight d = {d}")

    return A, B
