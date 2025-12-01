#!/usr/bin/env python
"""
Example 1: ILU Basics and Sparsity Visualization

Demonstrates:
- Creating a 3D finite difference matrix
- Computing ILU(0), ILU(k), and ILUT factorizations
- Visualizing sparsity patterns with spy plots
"""

import numpy as np
import matplotlib.pyplot as plt

from schurilu import ilu0, iluk, ilut
from schurilu.utils import fd3d


def main():
    # Create a 3D finite difference matrix
    nx, ny, nz = 8, 8, 8
    A = fd3d(nx, ny, nz)
    n = A.shape[0]
    print(f"Matrix size: {n} x {n}")
    print(f"Original nnz: {A.nnz}")

    # Compute different ILU factorizations
    result_ilu0 = ilu0(A)
    result_iluk = iluk(A, lfil=3)
    result_ilut = ilut(A, droptol=1e-3, lfil=20)

    print(f"\nILU(0) nnz: {result_ilu0.nnz}")
    print(f"ILU(3) nnz: {result_iluk.nnz}")
    print(f"ILUT nnz:   {result_ilut.nnz}")

    # Get complete L and U factors for visualization
    L0, U0 = result_ilu0.to_complete()
    Lk, Uk = result_iluk.to_complete()
    Lt, Ut = result_ilut.to_complete()

    # Spy plots
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    # Original matrix
    axes[0, 0].spy(A, markersize=0.5)
    axes[0, 0].set_title(f'A (nnz={A.nnz})')

    # ILU(0)
    axes[0, 1].spy(L0, markersize=0.5)
    axes[0, 1].set_title(f'ILU(0) L (nnz={L0.nnz})')
    axes[1, 1].spy(U0, markersize=0.5)
    axes[1, 1].set_title(f'ILU(0) U (nnz={U0.nnz})')

    # ILU(k)
    axes[0, 2].spy(Lk, markersize=0.5)
    axes[0, 2].set_title(f'ILU(3) L (nnz={Lk.nnz})')
    axes[1, 2].spy(Uk, markersize=0.5)
    axes[1, 2].set_title(f'ILU(3) U (nnz={Uk.nnz})')

    # ILUT
    axes[0, 3].spy(Lt, markersize=0.5)
    axes[0, 3].set_title(f'ILUT L (nnz={Lt.nnz})')
    axes[1, 3].spy(Ut, markersize=0.5)
    axes[1, 3].set_title(f'ILUT U (nnz={Ut.nnz})')

    axes[1, 0].axis('off')

    plt.tight_layout()
    plt.savefig('ilu_sparsity.png', dpi=150)
    plt.show()
    print("\nSaved: ilu_sparsity.png")


if __name__ == '__main__':
    main()

