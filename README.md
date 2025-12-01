# SchurILU

A NumPy/SciPy-based library for **incomplete LU preconditioners with explicit Schur complement support**, designed to provide a clean, reproducible ILU baseline for research in iterative solvers and learning-based preconditioning.

## Overview

SchurILU focuses on:

- Well-defined **ILU(0) / ILU(k) / ILUT** implementations
- **Partial / two-level ILU** on block-partitioned systems
- **GeMSLR**: General Multilevel Schur Low-Rank preconditioner
- Work with SciPy's Krylov solvers

This library is intended as a transparent, research-friendly reference implementation in pure Python. It is not designed to compete with production C/C++ libraries such as hypre or parGeMSLR, but rather to offer an accessible platform for algorithmic experimentation and prototyping.

> **Status: Alpha (v0.1.x)**
>
> This is early-stage research code. APIs may change between minor versions.
> Tested primarily on real SPD and general non-singular matrices.
> Bug reports and PRs are welcome -- see [Issues](https://github.com/Hitenze/SchurILU/issues).

## Installation

```bash
pip install schurilu
```

Or install from source:

```bash
git clone https://github.com/Hitenze/SchurILU.git
cd SchurILU
pip install -e .
```

For development (includes pytest):

```bash
pip install -e ".[dev]"
```

## Features

### ILU Factorizations

- **ilu0**: ILU(0) with no fill-in
- **iluk**: ILU(k) with level-based fill
- **ilut**: ILUT with threshold-based dropping

All factorizations support:
- Schur complement computation (partial factorization)
- float32, float64, complex64, complex128
- CSR sparse matrix input

### GeMSLR Preconditioner

- **GeMSLR**: General Multilevel Schur Low-Rank preconditioner
  - Multilevel domain decomposition via spectral graph partitioning
  - ILU factorization on interior blocks
  - Low-rank correction for Schur complement approximation
  - Arnoldi-based eigenvalue computation for optimal correction

> **Note on parallelism**: The GeMSLR algorithm is designed for parallel execution -- the interior blocks at each level are independent and can be factored/solved in parallel. This pure-Python implementation is sequential; for production HPC use, see [parGeMSLR](https://github.com/Hitenze/pargemslr).

### Reordering

- **multilevel_partition**: Spectral k-way partitioning for multilevel ordering
- **spectral_kway**: Recursive spectral bisection using Fiedler vector
- **unweighted_laplacian**: Graph Laplacian construction
- **connected_components**: Graph connectivity analysis

> **Note**: The built-in spectral partitioner is a naive implementation for research/prototyping. For better partition quality, consider using METIS and passing a custom permutation to `GeMSLR(A, p=your_perm, lev_ptr=your_lev_ptr)`.

### Krylov Solvers

- **fgmres**: Flexible GMRES (real)
- **fgmrez**: Flexible GMRES (complex)
- **pcg**: Preconditioned Conjugate Gradient
- **planczos**: Preconditioned Lanczos for eigenvalue estimation

## Usage

```python
import numpy as np
from schurilu import ilu0, iluk, ilut, fgmres, pcg
from schurilu.utils import fd3d

# Create a 3D Laplacian matrix (10x10x10 grid)
A = fd3d(10, 10, 10)

# ILU factorization
ilu = ilu0(A)

# Solve with preconditioned GMRES
b = np.ones(A.shape[0])
x, info = fgmres(A, b, M=ilu, tol=1e-10)

# Solve with preconditioned CG (for SPD matrices)
x, info = pcg(A, b, M=ilu, tol=1e-10)
```

### Schur Complement

```python
# Partial factorization with Schur complement
nB = 50  # Size of B block
result = ilu0(A, nB=nB)

# Access Schur complement
S = result.S  # Schur complement matrix
E = result.E  # Transformed lower-left block (E_orig * U_B^{-1})
F = result.F  # Transformed upper-right block (L_B^{-1} * F_orig)
```

### ILU with Fill-in Control

```python
# Level-based fill
result = iluk(A, lfil=2)  # Allow fill up to level 2

# Threshold-based dropping
result = ilut(A, droptol=1e-4, lfil=20)
```

### GeMSLR Preconditioner

```python
from schurilu import GeMSLR, fgmres

# Create GeMSLR preconditioner with automatic spectral partitioning
pre = GeMSLR(A, nlev=3, k=4, droptol=1e-3, rank_k=10)

# Solve with FGMRES
x, info = fgmres(A, b, M=pre, tol=1e-10)

# For exact factorization (rapid convergence with full rank)
pre_exact = GeMSLR(A, nlev=2, k=2, droptol=0.0, rank_k=50,
                   arnoldi_tol=1e-14, arnoldi_maxiter=100)
```

Key parameters:
- `nlev`: Number of levels in the multilevel hierarchy
- `k`: Number of partitions per level (must be power of 2)
- `droptol`: Drop tolerance for ILU (0 = exact)
- `rank_k`: Target rank for low-rank correction
- `theta`: Spectrum shift parameter (default 0)

## API Reference

### ILUResult

All ILU functions return an `ILUResult` object with:

**Attributes:**
- `L`: Lower triangular factor (n x n sparse matrix, unit diagonal not stored)
- `D`: Diagonal stored as inverse (1/d_ii). For partial factorization, only `D[:nB]` is valid.
- `U`: Upper triangular factor (n x n sparse matrix, diagonal not stored)
- `E`, `F`, `S`: Schur complement parts (only present if `nB < n`). Note: `E` stores `E_orig * U_B^{-1}` and `F` stores `L_B^{-1} * F_orig`, not the original blocks.
- `n`: Matrix dimension
- `nB`: Size of B block (`nB == n` for full factorization)

**Methods:**
- `solve(b)`: Apply preconditioner. For partial factorization, solves only the B-block; the C-block is passed through unchanged.
- `to_linear_operator()`: Convert to scipy LinearOperator (n x n).
- `to_complete()`: Returns `(L_complete, U_complete)` with diagonals restored. Returns (nB x nB) matrices containing only the B-block factors.

### GeMSLR

The `GeMSLR` class provides a multilevel Schur low-rank preconditioner:

**Attributes:**
- `n`: Matrix dimension
- `nlev`: Number of levels in hierarchy
- `p`: Permutation array
- `lev_ptr`: Level pointers

**Methods:**
- `solve(b)`: Apply preconditioner
- `to_linear_operator()`: Convert to scipy LinearOperator

**Properties:**
- `nnz`: Total nonzeros in preconditioner
- `nnz_ilu`: Nonzeros in ILU factors
- `nnz_lowrank`: Nonzeros in low-rank corrections
- `fill_factor()`: Ratio of preconditioner nnz to original matrix nnz

### arnoldi

Low-level Arnoldi iteration for computing dominant eigenvalues:

```python
V, H, m, iters = arnoldi(matvec, n, neig=5, tol=1e-12, maxiter=600)
```

**Parameters:**
- `matvec`: Callable for matrix-vector product
- `n`: Matrix dimension
- `neig`: Target number of eigenvalues
- `tol`: Convergence tolerance
- `maxiter`: Maximum iterations

**Returns:**
- `V`: Ritz vectors (n x m)
- `H`: Upper triangular Schur form (m x m)
- `m`: Number of converged eigenvalues
- `iters`: Total iterations performed

## Testing

```bash
pytest tests/ -v
```

## Examples

The `examples/` directory contains runnable scripts:

| Example | Description |
|---------|-------------|
| `01_ilu_basics.py` | ILU factorizations and sparsity visualization |
| `02_preconditioned_gmres.py` | FGMRES convergence with ILU preconditioners |
| `03_gemslr_preconditioner.py` | GeMSLR multilevel preconditioner comparison |

Run with:

```bash
pip install matplotlib  # for visualization
python examples/01_ilu_basics.py
```

## Related Publications

The design of SchurILU is informed by the following works:

- T. Xu, R. Li, and D. Osei-Kuffuor, "A Two-level GPU-Accelerated Incomplete LU Preconditioner for General Sparse Linear Systems," *International Journal of High Performance Computing Applications*, 39(3):424-442, 2025.

- T. Xu, V. Kalantzis, R. Li, Y. Xi, G. Dillon, and Y. Saad, "parGeMSLR: A Parallel Multilevel Schur Complement Low-Rank Preconditioning and Solution Package for General Sparse Matrices," *Parallel Computing*, 113:102956, 2022.

If you use SchurILU in academic work, please consider citing the relevant papers above. An official SchurILU citation will be provided once available.

## License

MIT License
