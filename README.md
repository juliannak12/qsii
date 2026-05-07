# qsii

**Quantum weight enumerators for stabilizer codes via tensor network contraction.**

Julianna Kelley · Virginia Tech · Advisor: Dr. ChunJun Cao · 2024–2026

---

## What this is

A stabilizer code's weight enumerator A(z) tells you everything about its error-correcting capability — the minimum non-zero coefficient gives you the code distance, which determines how many errors the code can catch. The problem is that computing A(z) is exponentially hard in general. This project uses the [Quantum LEGO tensor network framework](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.3.020332) (Cao & Lackey 2022) to do it efficiently.

The core idea: represent each stabilizer as a local tensor, connect them into a network matching the code's graph structure, and contract the network to read off A(z). For 1D codes this becomes a transfer matrix problem solvable in O(log n) time. I can compute enumerators for 1D cluster states with 1000+ qubits in under a second. 2D is harder — brute force works up to ~16 qubits, and row-by-row contraction is in progress.

What I find genuinely satisfying about this is that the same computation that spits out a polynomial also secretly encodes a classical statistical mechanics partition function. GHZ → 1D Ising. Toric code → 2D Ising on a torus, with a verified ratio D_Z(z)/Z_Ising(z) = 0.5000 (explained by global Z₂ spin-flip symmetry). The connection feels deep in a way I'm still working to fully understand.

---

## Code families

| Code | n | Method | Result |
|------|---|--------|--------|
| GHZ | any | Transfer matrix (parity-augmented) | A(z) = (1+z)^n |
| 1D cluster | any | Transfer matrix, 16×16, O(log n) | d=2, scales to 1000+ qubits |
| 2D cluster | ≤~16 | Brute force | exact, 3×4 verified |
| Toric code | 2L² | Brute force | D_Z/Z_Ising = 0.5000 ✓ |
| Surface code | 2L(L-1) | Brute force | D_Z/Z_Ising = 1.0000 ✓ |

---

## Enumerator types

**SL scalar A(z):** counts stabilizers by total Pauli weight. The main one.

**MacWilliams dual B(z):** the dual of A(z) via Krawtchouk polynomials (Shor & Laflamme 1997). Encodes the dual code structure.

**Double enumerator D_Z(z):** for CSS codes, restricts to the Z-sector. Maps directly to the Ising partition function.

**Complete enumerator D(W,X,Y,Z):** tracks each Pauli type separately. Most information, hardest to compute. Maps to the Ashkin-Teller model for the toric code (ratio = 0.250000 after correcting dual-lattice geometry).

---

## Quick start

```python
from qsii import compute_ghz_enumerator, compute_cluster_1d_enumerator

# GHZ — should give (1+z)^n exactly
A, B = compute_ghz_enumerator(n=8)
# A(z) = 1 + 8z + 28z^2 + 56z^3 + 70z^4 + 56z^5 + 28z^6 + 8z^7 + z^8

# 1D cluster — transfer matrix, O(log n)
A, B = compute_cluster_1d_enumerator(n=100)

# Toric code — with stat mech verification
from qsii import compute_toric_enumerators, verify_ising_ratio
A_SL, B_SL, D_Z, D_X = compute_toric_enumerators(L=2)
verify_ising_ratio(L=2)
# D_Z(z) / Z_Ising(z) = 0.5000 at all test points
```

---

## Structure

```
qsii/
├── core/
│   ├── polynomial.py     # Polynomial arithmetic (translated from MATLAB)
│   ├── macwilliams.py    # MacWilliams transform via Krawtchouk polynomials
│   └── brute_force.py    # Reference implementations for small n
└── codes/
    ├── ghz.py            # Parity-augmented transfer matrix
    ├── cluster.py        # 1D (transfer matrix) + 2D (brute force)
    └── toric_surface.py  # Toric + surface codes + Ising verification

scripts/demo.py           # Runs everything and saves figures
tests/test_all.py         # pytest suite against known analytic results
```

---

## Install

```bash
pip install numpy scipy matplotlib
# then just run from the repo root
python scripts/demo.py
```

---

## References

1. Cao & Lackey, *PRX Quantum* 3, 020332 (2022) — Quantum LEGO framework
2. Shor & Laflamme, *PRL* 78, 1600 (1997) — Quantum MacWilliams identity
3. Rains, *IEEE Trans. Inf. Theory* 44, 1388 (1998) — Weight enumerator theory
4. Kitaev, *Ann. Phys.* 303, 2–30 (2003) — Toric code
5. Dennis, Kitaev, Landahl & Preskill, *J. Math. Phys.* 43, 4452 (2002) — Topological memory
6. Raussendorf & Briegel, *PRL* 86, 5188 (2001) — Cluster states
