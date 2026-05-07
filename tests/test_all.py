"""
tests/test_all.py — unit tests against known analytic results.

Run with: pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from math import comb

from qsii.codes.ghz import compute_ghz_enumerator
from qsii.codes.cluster import compute_cluster_1d_enumerator, compute_cluster_2d_enumerator
from qsii.codes.toric_surface import compute_toric_enumerators, compute_surface_enumerators


class TestGHZ:
    def test_binomial_coefficients(self):
        """A(z) = (1+z)^n, so A_j = C(n,j)."""
        for n in [4, 6, 8]:
            A, _ = compute_ghz_enumerator(n, verbose=False)
            for j, c in enumerate(A.coeffs):
                assert abs(c - comb(n, j)) < 0.5, f"n={n} j={j}: got {c}"

    def test_partition_sum(self):
        """A(1) = 2^n."""
        for n in [4, 6, 8]:
            A, _ = compute_ghz_enumerator(n, verbose=False)
            assert abs(A.evaluate(1.0) - 2**n) < 1.0


class TestCluster1D:
    def test_partition_sum(self):
        """A(1) = 2^n."""
        for n in [4, 6, 8, 10]:
            A, _ = compute_cluster_1d_enumerator(n, verbose=False)
            assert abs(A.evaluate(1.0) - 2**n) < 1.0

    def test_A0_equals_1(self):
        """Only the identity has weight 0."""
        A, _ = compute_cluster_1d_enumerator(8, verbose=False)
        assert abs(A.coeffs[0] - 1.0) < 0.5

    def test_matches_brute_force(self):
        """Transfer matrix result matches brute force for small n."""
        from qsii.core.brute_force import weight_enumerator_brute_force
        from qsii.codes.cluster import cluster_1d_generators
        for n in [4, 6, 8]:
            A_tm, _ = compute_cluster_1d_enumerator(n, verbose=False)
            A_bf    = weight_enumerator_brute_force(cluster_1d_generators(n), n)
            for i in range(min(len(A_tm.coeffs), len(A_bf.coeffs))):
                assert abs(A_tm.coeffs[i] - A_bf.coeffs[i]) < 0.5


class TestCluster2D:
    def test_partition_sum(self):
        A, _ = compute_cluster_2d_enumerator(3, 3, verbose=False)
        assert abs(A.evaluate(1.0) - 2**9) < 1.0


class TestToricCode:
    def test_partition_sum(self):
        A, _, _, _ = compute_toric_enumerators(2, verbose=False)
        assert abs(A.evaluate(1.0) - 2**(2*4)) < 1.0

    def test_DZ_equals_DX(self):
        """By torus symmetry, D_Z = D_X."""
        _, _, DZ, DX = compute_toric_enumerators(2, verbose=False)
        for z in [0.1, 0.3, 0.5]:
            assert abs(DZ.evaluate(z) - DX.evaluate(z)) < 1e-6


class TestSurfaceCode:
    def test_A0_equals_1(self):
        A, _, _, _ = compute_surface_enumerators(3, verbose=False)
        assert abs(A.coeffs[0] - 1.0) < 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
