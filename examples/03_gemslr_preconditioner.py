#!/usr/bin/env python
"""
Example 3: GeMSLR Multilevel Preconditioner

Demonstrates:
- Building a GeMSLR preconditioner with spectral partitioning
- Comparing GeMSLR vs ILU for FGMRES convergence
- Visualizing the multilevel structure
"""

import numpy as np
import matplotlib.pyplot as plt

from schurilu import GeMSLR, ilu0, ilut, fgmres, multilevel_partition
from schurilu.utils import fd3d


def solve_with_history(A, b, M, label=""):
    """Solve and track residual history."""
    residuals = []

    def callback(r):
        residuals.append(r)

    x, info = fgmres(A, b, M=M, tol=1e-10, maxiter=300, restart=30, callback=callback)

    res_final = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    status = "converged" if info == 0 else f"info={info}"
    print(f"{label:20s}: {len(residuals):3d} iters, residual = {res_final:.2e} ({status})")

    return residuals


def main():
    # Create problem: 3D Laplacian with diagonal shift (mildly indefinite)
    nx, ny, nz = 10, 10, 10
    shift = 0.2 # Makes problem harder
    A = fd3d(nx, ny, nz, shift=shift)
    n = A.shape[0]

    np.random.seed(42)
    x_true = np.random.randn(n)
    b = A @ x_true

    print(f"Problem size: {n}")
    print(f"Matrix nnz: {A.nnz}")
    print(f"Diagonal shift: {shift} (mildly indefinite)")
    print("=" * 60)

    # Build preconditioners
    print("\nBuilding preconditioners...")

    # ILU(0) baseline
    M_ilu0 = ilu0(A)
    print(f"  ILU(0): nnz = {M_ilu0.nnz}, fill = {M_ilu0.nnz / A.nnz:.2f}x")

    # ILUT baseline
    M_ilut = ilut(A, droptol=1e-3, lfil=20)
    print(f"  ILUT:   nnz = {M_ilut.nnz}, fill = {M_ilut.nnz / A.nnz:.2f}x")

    # GeMSLR with different configurations
    M_gemslr_no_lr = GeMSLR(A, nlev=3, k=4, droptol=1e-3, rank_k=0)
    print(f"  GeMSLR (no LR): nnz = {M_gemslr_no_lr.nnz}, fill = {M_gemslr_no_lr.fill_factor():.2f}x")

    M_gemslr = GeMSLR(A, nlev=3, k=4, droptol=1e-3, rank_k=10)
    print(f"  GeMSLR (rank=10): nnz = {M_gemslr.nnz}, fill = {M_gemslr.fill_factor():.2f}x")
    print(f"    - ILU nnz: {M_gemslr.nnz_ilu}, LowRank nnz: {M_gemslr.nnz_lowrank}")

    print("=" * 60)
    print("\nSolving with FGMRES...")

    # Solve with different preconditioners
    res_ilu0 = solve_with_history(A, b, M_ilu0, "ILU(0)")
    res_ilut = solve_with_history(A, b, M_ilut, "ILUT")
    res_gemslr_no_lr = solve_with_history(A, b, M_gemslr_no_lr, "GeMSLR (no LR)")
    res_gemslr = solve_with_history(A, b, M_gemslr, "GeMSLR (rank=10)")

    # Plot convergence comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Convergence plot
    ax1 = axes[0]
    ax1.semilogy(res_ilu0, 'b-', label='ILU(0)', linewidth=2)
    ax1.semilogy(res_ilut, 'g-', label='ILUT', linewidth=2)
    ax1.semilogy(res_gemslr_no_lr, 'r--', label='GeMSLR (no LR)', linewidth=2)
    ax1.semilogy(res_gemslr, 'r-', label='GeMSLR (rank=10)', linewidth=2)

    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Relative Residual', fontsize=12)
    ax1.set_title('FGMRES Convergence Comparison', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Visualize permuted matrix structure
    ax2 = axes[1]
    p, lev_ptr, nlev = multilevel_partition(A, nlev=3, k=4)
    Ap = A[p, :][:, p]

    ax2.spy(Ap, markersize=0.3)
    ax2.set_title('Permuted Matrix (Multilevel Structure)', fontsize=14)

    # Draw level boundaries
    for ptr in lev_ptr[1:-1]:
        ax2.axhline(y=ptr - 0.5, color='red', linewidth=1, linestyle='--')
        ax2.axvline(x=ptr - 0.5, color='red', linewidth=1, linestyle='--')

    plt.tight_layout()
    plt.savefig('gemslr_comparison.png', dpi=150)
    plt.show()
    print("\nSaved: gemslr_comparison.png")


if __name__ == '__main__':
    main()

