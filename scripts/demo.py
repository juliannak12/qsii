#!/usr/bin/env python3
"""
demo.py — run all enumerators and produce figures.

Usage: python scripts/demo.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qsii.codes.ghz import compute_ghz_enumerator
from qsii.codes.cluster import compute_cluster_1d_enumerator, compute_cluster_2d_enumerator
from qsii.codes.toric_surface import (
    compute_toric_enumerators, compute_surface_enumerators, verify_ising_ratio
)

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGS, exist_ok=True)

MAROON = "#861F41"
ORANGE = "#E5751F"
TEAL   = "#00808E"
NAVY   = "#1A1A2E"

def bar_plot(poly, title, fname, color=MAROON):
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
    weights = list(range(len(poly.coeffs)))
    counts  = [round(c) if abs(c - round(c)) < 0.1 else c for c in poly.coeffs]
    ax.bar(weights, counts, color=color, edgecolor=NAVY, linewidth=0.6, alpha=0.88)
    for w, c in zip(weights, counts):
        if abs(c) > 0:
            ax.text(w, c + max(max(counts), 1)*0.02, str(int(c)), ha='center', fontsize=9)
    ax.set_xlabel('Pauli weight', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_facecolor('white')
    plt.tight_layout()
    path = os.path.join(FIGS, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")
    return path


def compare_plot(polys, labels, title, fname, colors=None):
    colors = colors or [MAROON, ORANGE, TEAL, NAVY]
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor='white')
    max_n = max(len(p.coeffs) for p in polys)
    x = np.arange(max_n)
    width = 0.75 / len(polys)
    for k, (poly, label, color) in enumerate(zip(polys, labels, colors)):
        vals = [round(poly.coeffs[i]) if i < len(poly.coeffs) else 0 for i in range(max_n)]
        offset = (k - len(polys)/2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85, edgecolor=NAVY, linewidth=0.4)
    ax.set_xlabel('Weight', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend()
    ax.set_facecolor('white')
    plt.tight_layout()
    path = os.path.join(FIGS, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")
    return path


print("=" * 55)
print("QSII Weight Enumerators  ·  Julianna Kelley")
print("=" * 55)

print("\n── GHZ ──")
A_ghz6, B_ghz6 = compute_ghz_enumerator(6)
A_ghz8, _      = compute_ghz_enumerator(8, verbose=False)
bar_plot(A_ghz6, "GHZ (n=6):  A(z) = (1+z)⁶", "ghz_6.png")
compare_plot([A_ghz6, A_ghz8], ["n=6", "n=8"],
             "GHZ states: A(z) comparison", "ghz_compare.png")

print("\n── 1D Cluster ──")
A_c1, B_c1 = compute_cluster_1d_enumerator(8)
A_c1_12, _ = compute_cluster_1d_enumerator(12, verbose=False)
bar_plot(A_c1, "1D Cluster (n=8):  A(z)", "cluster_1d_8.png", ORANGE)
compare_plot([A_c1, B_c1], ["A(z) primal", "B(z) dual"],
             "1D Cluster (n=8): MacWilliams pair", "cluster_1d_macwilliams.png",
             [ORANGE, TEAL])

print("\n── 2D Cluster ──")
A_c2, _ = compute_cluster_2d_enumerator(3, 4)
bar_plot(A_c2, "2D Cluster (3×4, n=12):  A(z)", "cluster_2d_3x4.png", TEAL)

print("\n── Toric Code ──")
A_tor, B_tor, DZ_tor, DX_tor = compute_toric_enumerators(2)
compare_plot([DZ_tor, DX_tor], ["D_Z(z)", "D_X(z)"],
             "Toric Code L=2: double enumerators", "toric_double.png",
             [MAROON, ORANGE])
compare_plot([A_tor, B_tor], ["A(z)", "B(z)"],
             "Toric Code L=2: MacWilliams pair", "toric_macwilliams.png",
             [MAROON, TEAL])

print("\n── Surface Code ──")
A_surf, _, DZ_surf, DX_surf = compute_surface_enumerators(3)
compare_plot([DZ_surf, DX_surf], ["D_Z(z)", "D_X(z)"],
             "Surface Code L=3: double enumerators", "surface_double.png",
             [MAROON, ORANGE])

print("\n── Ising Ratio Verification ──")
verify_ising_ratio(L=2)

print("\n── All families ──")
compare_plot(
    [A_ghz6, A_c1, A_c2, A_tor],
    ["GHZ n=6", "1D Cluster n=8", "2D Cluster 3×4", "Toric L=2"],
    "All code families: SL enumerators",
    "all_families.png"
)

print("\nDone.")
