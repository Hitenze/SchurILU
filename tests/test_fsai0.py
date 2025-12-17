"""
Tests for FSAI0 preconditioner built from matrix pattern.
"""

import numpy as np
import pytest
from scipy import sparse

from schurilu.utils import fd3d


def _fsai0():
    from schurilu.preconditioners import fsai0
    return fsai0


class TestFSAI0Structure:
    def test_pattern_matches_A_and_diag_last(self):
        A = fd3d(4, 4, 1)
        fsai0 = _fsai0()
        pre = fsai0(A)
        L = pre.L

        assert sparse.isspmatrix_csr(L)
        assert L.shape == A.shape

        n = A.shape[0]
        for i in range(n):
            a_start, a_end = A.indptr[i], A.indptr[i + 1]
            a_cols = A.indices[a_start:a_end]
            # Remove diagonal from A row pattern
            a_cols_wo_diag = [c for c in a_cols if c != i]

            l_start, l_end = L.indptr[i], L.indptr[i + 1]
            l_cols = L.indices[l_start:l_end]
            # Diagonal should be last entry
            assert l_cols[-1] == i
            # Off-diagonal pattern should match A's off-diagonal
            l_off = l_cols[:-1]
            assert set(l_off) == set(a_cols_wo_diag)

    def test_dtype_respected(self):
        A = fd3d(3, 3, 1, dtype=np.float32)
        fsai0 = _fsai0()
        pre = fsai0(A)
        assert pre.L.dtype == np.float32


class TestFSAI0Properties:
    def test_exact_for_diagonal_matrix(self):
        rng = np.random.default_rng(0)
        d = 1.0 + rng.random(8)
        A = sparse.diags(d, format="csr")
        fsai0 = _fsai0()
        pre = fsai0(A)

        L = pre.L
        M = (L.T @ L).toarray()
        I_approx = A @ M
        np.testing.assert_allclose(I_approx.toarray() if sparse.issparse(I_approx) else I_approx, np.eye(A.shape[0]), rtol=0, atol=1e-12)

    def test_M_is_spd(self):
        A = fd3d(4, 4, 1)
        fsai0 = _fsai0()
        pre = fsai0(A)
        L = pre.L
        M = (L.T @ L).toarray()
        eigs = np.linalg.eigvalsh(M)
        assert np.all(eigs > 0)

    def test_apply_matches_linear_operator(self):
        A = fd3d(4, 4, 1)
        fsai0 = _fsai0()
        pre = fsai0(A)
        M = pre.to_linear_operator()
        rng = np.random.default_rng(123)
        b = rng.standard_normal(A.shape[0])
        y1 = pre.solve(b)
        y2 = M @ b
        np.testing.assert_allclose(y1, y2, rtol=0, atol=1e-14)

    def test_tiny_diagonal_is_guarded(self):
        # Diagonal matrix with extremely small positive entries
        n = 6
        d = np.full(n, 1e-300)
        A = sparse.diags(d, format="csr")
        fsai0 = _fsai0()
        pre = fsai0(A)
        # No inf or nan in L
        assert np.isfinite(pre.L.data).all()
        # M = L^T L should be SPD
        M = (pre.L.T @ pre.L).toarray()
        w = np.linalg.eigvalsh(M)
        assert np.all(w > 0)
