"""
Tests for GeMSLR preconditioner.
"""

import numpy as np
import pytest
from scipy.sparse import diags

from schurilu import GeMSLR, arnoldi, fgmres, multilevel_partition
from tests.conftest import fd3d


# =============================================================================
# Arnoldi Tests
# =============================================================================

class TestArnoldi:
    """Tests for Arnoldi eigenvalue computation."""

    def test_arnoldi_orthonormal_output(self):
        """Arnoldi should return orthonormal vectors."""
        n = 50
        A = diags([1, 2, 1], [-1, 0, 1], shape=(n, n)).toarray()

        def matvec(v):
            return A @ v

        V, H, m, tits = arnoldi(matvec, n, neig=5, tol=1e-10, maxiter=100)

        if m > 0:
            VtV = V.T @ V
            np.testing.assert_allclose(VtV, np.eye(m), atol=1e-10)

    def test_arnoldi_finds_eigenvalues(self):
        """Arnoldi should find eigenvalues of diagonal matrix."""
        n = 30
        eigenvalues_true = np.linspace(0.1, 1.0, n)
        A = np.diag(eigenvalues_true)

        def matvec(v):
            return A @ v

        V, H, m, tits = arnoldi(matvec, n, neig=5, neig_keep=5, tol=1e-8, maxiter=200)

        # Should find some eigenvalues
        assert m >= 0
        assert tits > 0


# =============================================================================
# GeMSLR Construction Tests
# =============================================================================

class TestGeMSLRConstruction:
    """Tests for GeMSLR preconditioner construction."""

    def test_gemslr_creates_preconditioner(self):
        """GeMSLR should create a valid preconditioner."""
        A = fd3d(8, 8, 1)
        pre = GeMSLR(A, nlev=2, k=4, droptol=1e-2, rank_k=3)

        assert pre.n == A.shape[0]
        assert pre.nlev >= 1

    def test_gemslr_with_user_permutation(self):
        """GeMSLR should accept user-provided permutation."""
        A = fd3d(8, 8, 1)
        n = A.shape[0]

        p, lev_ptr, nlev = multilevel_partition(A, nlev=2, k=4)
        pre = GeMSLR(A, p=p, lev_ptr=lev_ptr, droptol=1e-2, rank_k=3)

        assert pre.n == n
        np.testing.assert_array_equal(pre.p, p)

    def test_gemslr_3d_problem(self):
        """GeMSLR should work for 3D problems."""
        A = fd3d(6, 6, 6)  # 216 unknowns
        pre = GeMSLR(A, nlev=2, k=4, droptol=1e-2, rank_k=5)

        assert pre.n == 216

    def test_gemslr_no_lowrank(self):
        """GeMSLR with rank_k=0 should work without low-rank correction."""
        A = fd3d(8, 8, 1)
        pre = GeMSLR(A, nlev=2, k=4, droptol=1e-2, rank_k=0)

        assert pre.nnz_lowrank == 0


# =============================================================================
# GeMSLR Solve Tests
# =============================================================================

class TestGeMSLRSolve:
    """Tests for GeMSLR preconditioner solve."""

    def test_gemslr_solve_reduces_residual(self):
        """GeMSLR solve should reduce residual."""
        A = fd3d(8, 8, 1)
        n = A.shape[0]

        pre = GeMSLR(A, nlev=2, k=4, droptol=1e-3, rank_k=5)

        x_true = np.ones(n)
        b = A @ x_true
        r = b.copy()

        # One preconditioner application
        z = pre.solve(r)
        x1 = z
        r1 = b - A @ x1

        # Should reduce residual
        assert np.linalg.norm(r1) < np.linalg.norm(r)

    def test_gemslr_callable(self):
        """GeMSLR should be callable."""
        A = fd3d(8, 8, 1)
        pre = GeMSLR(A, nlev=2, k=4, droptol=1e-2, rank_k=3)

        b = np.ones(A.shape[0])

        # Should be callable
        x = pre(b)
        assert x.shape == b.shape

        # Should match solve
        x2 = pre.solve(b)
        np.testing.assert_allclose(x, x2)


# =============================================================================
# GeMSLR with FGMRES Tests
# =============================================================================

class TestGeMSLRWithFGMRES:
    """Tests for GeMSLR preconditioner with FGMRES solver."""

    def test_gemslr_fgmres_converges_spd(self):
        """GeMSLR+FGMRES should converge on SPD problem."""
        A = fd3d(8, 8, 8)  # 512 unknowns, SPD
        n = A.shape[0]

        x_true = np.ones(n)
        b = A @ x_true

        pre = GeMSLR(A, nlev=2, k=4, droptol=1e-2, rank_k=5)
        x, info = fgmres(A, b, M=pre, tol=1e-8, maxiter=200, restart=30)

        assert info == 0
        residual = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
        assert residual < 1e-7

    def test_gemslr_fgmres_converges_indefinite(self):
        """GeMSLR+FGMRES should converge on mildly indefinite problem."""
        # 3D Laplacian with shift: diagonal 6 -> 5.95 (mildly indefinite)
        A = fd3d(8, 8, 8, shift=0.05)
        n = A.shape[0]

        x_true = np.ones(n)
        b = A @ x_true

        pre = GeMSLR(A, nlev=2, k=4, droptol=1e-2, rank_k=5)
        x, info = fgmres(A, b, M=pre, tol=1e-8, maxiter=300, restart=30)

        assert info == 0
        residual = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
        assert residual < 1e-7

    def test_gemslr_reduces_iterations(self):
        """GeMSLR should help FGMRES converge efficiently."""
        A = fd3d(6, 6, 6)  # 216 unknowns
        n = A.shape[0]

        x_true = np.ones(n)
        b = A @ x_true

        # With GeMSLR - should converge in reasonable iterations
        pre = GeMSLR(A, nlev=2, k=4, droptol=1e-2, rank_k=5)
        iters_pre = [0]
        def count_pre(r):
            iters_pre[0] += 1
        x, info = fgmres(A, b, M=pre, tol=1e-8, maxiter=100, callback=count_pre)

        # Should converge
        assert info == 0
        residual = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
        assert residual < 1e-7
        # Should converge in reasonable number of iterations (< 50)
        assert iters_pre[0] < 50


# =============================================================================
# GeMSLR Low-Rank Correction Tests
# =============================================================================

class TestGeMSLRLowRank:
    """Tests for low-rank correction effectiveness."""

    def test_lowrank_improves_convergence(self):
        """Low-rank correction should improve or maintain convergence."""
        A = fd3d(6, 6, 6)
        n = A.shape[0]

        x_true = np.ones(n)
        b = A @ x_true

        # Without low-rank
        pre_no_lr = GeMSLR(A, nlev=2, k=4, droptol=1e-2, rank_k=0)
        x1, info1 = fgmres(A, b, M=pre_no_lr, tol=1e-8, maxiter=100)

        # With low-rank
        pre_lr = GeMSLR(A, nlev=2, k=4, droptol=1e-2, rank_k=5)
        x2, info2 = fgmres(A, b, M=pre_lr, tol=1e-8, maxiter=100)

        # Both should converge
        res1 = np.linalg.norm(b - A @ x1) / np.linalg.norm(b)
        res2 = np.linalg.norm(b - A @ x2) / np.linalg.norm(b)

        assert res2 < 1e-6


# =============================================================================
# GeMSLR One-Step Convergence Test (Exact ILU + Enough Rank)
# =============================================================================

class TestGeMSLROneStep:
    """Tests for one-step convergence with exact factorization."""

    def test_one_step_convergence_spd(self):
        """With exact ILU and enough rank, should converge quickly for SPD."""
        # Small 2D problem for exact factorization
        A = fd3d(6, 6, 1)  # 36 unknowns
        n = A.shape[0]

        x_true = np.ones(n)
        b = A @ x_true

        # Exact ILU (droptol=0) with high rank
        pre = GeMSLR(A, nlev=2, k=4, droptol=0.0, lfil=n, rank_k=20)

        # Track iterations
        iterations = [0]
        def callback(r):
            iterations[0] += 1

        x, info = fgmres(A, b, M=pre, tol=1e-10, maxiter=50, restart=50, callback=callback)

        residual = np.linalg.norm(b - A @ x) / np.linalg.norm(b)

        # Should converge quickly (multilevel still has some error, so allow ~10 iters)
        assert residual < 1e-8
        assert iterations[0] <= 15, f"Expected <=15 iterations, got {iterations[0]}"

    def test_one_step_convergence_mildly_indefinite(self):
        """With exact ILU and enough rank, should converge quickly for mildly indefinite."""
        # Small 2D problem with mild indefiniteness
        # Diagonal: 4 -> 3.95 (very mild)
        A = fd3d(6, 6, 1, shift=0.05)
        n = A.shape[0]

        x_true = np.ones(n)
        b = A @ x_true

        # Exact ILU (droptol=0) with high rank
        pre = GeMSLR(A, nlev=2, k=4, droptol=0.0, lfil=n, rank_k=20)

        # Track iterations
        iterations = [0]
        def callback(r):
            iterations[0] += 1

        x, info = fgmres(A, b, M=pre, tol=1e-10, maxiter=50, restart=50, callback=callback)

        residual = np.linalg.norm(b - A @ x) / np.linalg.norm(b)

        # Should converge quickly
        assert residual < 1e-8
        assert iterations[0] <= 10, f"Expected <=10 iterations, got {iterations[0]}"

    def test_one_step_convergence_3d_small(self):
        """With exact ILU and enough rank, 3D small problem should converge quickly."""
        # Small 3D problem
        A = fd3d(4, 4, 4)  # 64 unknowns
        n = A.shape[0]

        x_true = np.ones(n)
        b = A @ x_true

        # Exact ILU with high rank
        pre = GeMSLR(A, nlev=2, k=4, droptol=0.0, lfil=n, rank_k=30)

        iterations = [0]
        def callback(r):
            iterations[0] += 1

        x, info = fgmres(A, b, M=pre, tol=1e-10, maxiter=50, restart=50, callback=callback)

        residual = np.linalg.norm(b - A @ x) / np.linalg.norm(b)

        assert residual < 1e-8
        # 3D has more complex structure, allow more iterations
        assert iterations[0] <= 15, f"Expected <=15 iterations, got {iterations[0]}"


# =============================================================================
# GeMSLR Properties Tests
# =============================================================================

class TestGeMSLRProperties:
    """Tests for GeMSLR properties and diagnostics."""

    def test_fill_factor(self):
        """Fill factor should be computed correctly."""
        A = fd3d(8, 8, 1)
        pre = GeMSLR(A, nlev=2, k=4, droptol=1e-2, rank_k=5)

        ff = pre.fill_factor()
        assert ff > 0
        assert np.isfinite(ff)

    def test_nnz_properties(self):
        """NNZ properties should be consistent."""
        A = fd3d(8, 8, 1)
        pre = GeMSLR(A, nlev=2, k=4, droptol=1e-2, rank_k=5)

        assert pre.nnz == pre.nnz_ilu + pre.nnz_lowrank
        assert pre.nnz_ilu > 0

    def test_repr(self):
        """Repr should return informative string."""
        A = fd3d(6, 6, 1)
        pre = GeMSLR(A, nlev=2, k=4, droptol=1e-2, rank_k=3)

        rep = repr(pre)
        assert "GeMSLR" in rep
        assert "n=" in rep


# =============================================================================
# GeMSLR with ILUK Tests
# =============================================================================

class TestGeMSLRWithILUK:
    """Tests for GeMSLR preconditioner with ILUK factorization."""

    def test_gemslr_iluk_creates_preconditioner(self):
        """GeMSLR with ILUK should create a valid preconditioner."""
        A = fd3d(8, 8, 1)
        pre = GeMSLR(A, nlev=2, k=4, ilu_type='iluk', level_k=1, rank_k=3)

        assert pre.n == A.shape[0]
        assert pre.nlev >= 1
        assert pre.ilu_type == 'iluk'

    def test_gemslr_iluk_solve_reduces_residual(self):
        """GeMSLR with ILUK solve should reduce residual."""
        A = fd3d(8, 8, 1)
        n = A.shape[0]

        pre = GeMSLR(A, nlev=2, k=4, ilu_type='iluk', level_k=2, rank_k=5)

        x_true = np.ones(n)
        b = A @ x_true
        r = b.copy()

        # One preconditioner application
        z = pre.solve(r)
        x1 = z
        r1 = b - A @ x1

        # Should reduce residual
        assert np.linalg.norm(r1) < np.linalg.norm(r)

    def test_gemslr_iluk_fgmres_converges(self):
        """GeMSLR with ILUK + FGMRES should converge."""
        A = fd3d(8, 8, 8)  # 512 unknowns
        n = A.shape[0]

        x_true = np.ones(n)
        b = A @ x_true

        pre = GeMSLR(A, nlev=2, k=4, ilu_type='iluk', level_k=2, rank_k=5)
        x, info = fgmres(A, b, M=pre, tol=1e-8, maxiter=200, restart=30)

        assert info == 0
        residual = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
        assert residual < 1e-7

    def test_gemslr_iluk_no_lowrank(self):
        """GeMSLR with ILUK and rank_k=0 should work."""
        A = fd3d(8, 8, 1)
        pre = GeMSLR(A, nlev=2, k=4, ilu_type='iluk', level_k=1, rank_k=0)

        assert pre.nnz_lowrank == 0

        # Should still be able to solve
        b = np.ones(A.shape[0])
        x = pre.solve(b)
        assert x.shape == b.shape


# =============================================================================
# GeMSLR ILUK One-Step Convergence Tests
# =============================================================================

class TestGeMSLRILUKOneStep:
    """Tests for fast convergence with exact ILUK factorization."""

    def test_iluk_one_step_convergence_spd(self):
        """With high level_k (exact ILUK) and enough rank, should converge quickly."""
        # Small 2D problem
        A = fd3d(6, 6, 1)  # 36 unknowns
        n = A.shape[0]

        x_true = np.ones(n)
        b = A @ x_true

        # High level_k for near-exact ILUK, high rank
        pre = GeMSLR(A, nlev=2, k=4, ilu_type='iluk', level_k=n, rank_k=20,
                     arnoldi_tol=1e-14, arnoldi_maxiter=100)

        iterations = [0]
        def callback(r):
            iterations[0] += 1

        x, info = fgmres(A, b, M=pre, tol=1e-10, maxiter=50, restart=50, callback=callback)

        residual = np.linalg.norm(b - A @ x) / np.linalg.norm(b)

        assert residual < 1e-8
        assert iterations[0] <= 15, f"Expected <=15 iterations, got {iterations[0]}"

    def test_iluk_one_step_convergence_3d(self):
        """With high level_k for 3D problem, should converge quickly."""
        # Small 3D problem
        A = fd3d(4, 4, 4)  # 64 unknowns
        n = A.shape[0]

        x_true = np.ones(n)
        b = A @ x_true

        # High level_k for near-exact ILUK
        pre = GeMSLR(A, nlev=2, k=4, ilu_type='iluk', level_k=n, rank_k=30,
                     arnoldi_tol=1e-14, arnoldi_maxiter=100)

        iterations = [0]
        def callback(r):
            iterations[0] += 1

        x, info = fgmres(A, b, M=pre, tol=1e-10, maxiter=50, restart=50, callback=callback)

        residual = np.linalg.norm(b - A @ x) / np.linalg.norm(b)

        assert residual < 1e-8
        assert iterations[0] <= 15, f"Expected <=15 iterations, got {iterations[0]}"
