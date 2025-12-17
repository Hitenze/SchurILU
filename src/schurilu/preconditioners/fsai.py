from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import cholesky, solve_triangular, LinAlgError


class FSAI0Result:
    def __init__(self, L: csr_matrix) -> None:
        self.L = L.tocsr()
        self.n = L.shape[0]
        self.dtype = L.dtype

    def solve(self, b: np.ndarray) -> np.ndarray:
        x = np.asarray(b, dtype=self.dtype).reshape(-1)
        y = self.L @ x
        z = self.L.T @ y
        return np.asarray(z)

    def __call__(self, b: np.ndarray) -> np.ndarray:
        return self.solve(b)

    def to_linear_operator(self) -> LinearOperator:
        return LinearOperator(shape=(self.n, self.n), matvec=self.solve, dtype=self.dtype)

    @property
    def nnz(self) -> int:
        return int(self.L.nnz)

    def __repr__(self) -> str:
        return f"FSAI0Result(n={self.n}, nnz(L)={self.L.nnz}, dtype={self.dtype})"


def _row_block(A: csr_matrix, row: int) -> tuple[np.ndarray, np.ndarray, float]:
    start, end = A.indptr[row], A.indptr[row + 1]
    cols = A.indices[start:end]
    vals = A.data[start:end]
    mask_off = cols != row
    Pi = cols[mask_off]
    # For SPD A, A[Pi,i] == A[i,Pi]; use row i values directly
    K_rhs = vals[mask_off].astype(A.dtype, copy=False)
    # diagonal
    diag = 0.0
    pos = np.where(cols == row)[0]
    if pos.size:
        diag = vals[int(pos[0])]
    return Pi, K_rhs, float(diag)


def fsai0(A: csr_matrix) -> FSAI0Result:
    if not sparse.isspmatrix_csr(A):
        A = A.tocsr()
    A = A.tocsr()
    n = A.shape[0]
    dtype = A.dtype
    zero_tol = 1e-14 if (np.issubdtype(dtype, np.floating) and dtype == np.float64) else 1e-6

    indptr = np.empty(n + 1, dtype=np.int32)
    indices_list: list[int] = []
    data_list: list[float] = []
    indptr[0] = 0

    for i in range(n):
        Pi, K_rhs, K_ii = _row_block(A, i)
        k = Pi.size

        if k > 0:
            # Build K_a = A[Pi, Pi]
            Ka = A[Pi][:, Pi].toarray()
            # Solve Ka y = K_rhs via Cholesky with a small ridge guard
            try:
                Lc = cholesky(Ka, lower=True, overwrite_a=False, check_finite=False)
            except LinAlgError:
                tau = max(zero_tol, zero_tol * float(np.max(np.diag(Ka))))
                Lc = cholesky(Ka + tau * np.eye(Ka.shape[0], dtype=Ka.dtype), lower=True, overwrite_a=False, check_finite=False)
            y = solve_triangular(Lc, K_rhs, lower=True, trans='N', check_finite=False)
            y = solve_triangular(Lc, y, lower=True, trans='T', check_finite=False)
            S = K_ii - float(y @ K_rhs)
            if S <= zero_tol or not np.isfinite(S):
                S = zero_tol
            d = 1.0 / np.sqrt(S)
            off_vals = (-d) * y
            diag_val = d
            # Append off-diagonals then diagonal (diag last)
            indices_list.extend(Pi.tolist())
            data_list.extend(off_vals.tolist())
            indices_list.append(i)
            data_list.append(diag_val)
        else:
            # Only diagonal
            kii_eff = K_ii if K_ii > zero_tol and np.isfinite(K_ii) else zero_tol
            d = 1.0 / np.sqrt(kii_eff)
            indices_list.append(i)
            data_list.append(d)

        indptr[i + 1] = len(indices_list)

    L = csr_matrix((np.asarray(data_list, dtype=dtype), np.asarray(indices_list, dtype=np.int32), indptr), shape=(n, n))
    L.eliminate_zeros()
    return FSAI0Result(L)
