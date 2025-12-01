#!/usr/bin/env python
"""
Example 2: Preconditioned GMRES with ILU

Demonstrates:
- Solving a linear system with FGMRES
- Comparing convergence: no preconditioner vs ILU(0) vs ILU(k) vs ILUT
- Plotting convergence history
"""

import numpy as np
import matplotlib.pyplot as plt

from schurilu import ilu0, iluk, ilut, fgmres
from schurilu.utils import fd3d


def solve_with_history(A, b, M=None, label=""):
    """Solve and track residual history."""
    residuals = []

    def callback(r):
        residuals.append(r)

    x, info = fgmres(A, b, M=M, tol=1e-10, maxiter=200, restart=30, callback=callback)

    res_final = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    print(f"{label:20s}: {len(residuals):3d} iters, final residual = {res_final:.2e}")

    return residuals


def main():
    # Create problem
    nx, ny, nz = 10, 10, 10
    A = fd3d(nx, ny, nz)
    n = A.shape[0]

    np.random.seed(42)
    x_true = np.random.randn(n)
    b = A @ x_true

    print(f"Problem size: {n}")
    print(f"Matrix nnz: {A.nnz}")
    print("-" * 50)

    # Build preconditioners
    M_ilu0 = ilu0(A)
    M_iluk = iluk(A, lfil=2)
    M_ilut = ilut(A, droptol=1e-3, lfil=15)

    print(f"ILU(0) fill factor: {M_ilu0.nnz / A.nnz:.2f}")
    print(f"ILU(2) fill factor: {M_iluk.nnz / A.nnz:.2f}")
    print(f"ILUT fill factor:   {M_ilut.nnz / A.nnz:.2f}")
    print("-" * 50)

    # Solve with different preconditioners
    res_none = solve_with_history(A, b, M=None, label="No preconditioner")
    res_ilu0 = solve_with_history(A, b, M=M_ilu0, label="ILU(0)")
    res_iluk = solve_with_history(A, b, M=M_iluk, label="ILU(2)")
    res_ilut = solve_with_history(A, b, M=M_ilut, label="ILUT")

    # Plot convergence
    plt.figure(figsize=(8, 6))
    plt.semilogy(res_none, 'k-', label='No preconditioner', linewidth=2)
    plt.semilogy(res_ilu0, 'b-', label='ILU(0)', linewidth=2)
    plt.semilogy(res_iluk, 'g-', label='ILU(2)', linewidth=2)
    plt.semilogy(res_ilut, 'r-', label='ILUT', linewidth=2)

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Relative Residual', fontsize=12)
    plt.title(f'FGMRES Convergence ({nx}x{ny}x{nz} 3D Laplacian)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig('gmres_convergence.png', dpi=150)
    plt.show()
    print("\nSaved: gmres_convergence.png")


if __name__ == '__main__':
    main()

