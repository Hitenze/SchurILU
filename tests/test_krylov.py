"""
Tests for Krylov solvers (fgmres, fgmrez, pcg, planczos).
"""

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator

from schurilu import fgmres, fgmrez, pcg, planczos, ilu0
from tests.conftest import fd3d, tridiag


# =============================================================================
# FGMRES Tests
# =============================================================================

class TestFGMRES:
    """Tests for FGMRES solver."""

    def test_fgmres_converges(self):
        """FGMRES should converge for well-conditioned system."""
        A = fd3d(5, 5, 1)
        n = A.shape[0]
        x_true = np.random.randn(n)
        b = A @ x_true

        x, info = fgmres(A, b, tol=1e-10, maxiter=100)

        assert info == 0
        np.testing.assert_allclose(x, x_true, rtol=1e-8)

    def test_fgmres_with_preconditioner(self):
        """FGMRES should work with ILU preconditioner."""
        A = fd3d(5, 5, 1)
        n = A.shape[0]
        x_true = np.random.randn(n)
        b = A @ x_true

        M = ilu0(A)
        x, info = fgmres(A, b, M=M, tol=1e-10, maxiter=50)

        assert info == 0
        np.testing.assert_allclose(x, x_true, rtol=1e-8)

    def test_fgmres_callable_preconditioner(self):
        """FGMRES should work with callable preconditioner."""
        A = fd3d(5, 5, 1)
        n = A.shape[0]
        x_true = np.random.randn(n)
        b = A @ x_true

        ilu = ilu0(A)
        x, info = fgmres(A, b, M=lambda v: ilu.solve(v), tol=1e-10, maxiter=50)

        assert info == 0
        np.testing.assert_allclose(x, x_true, rtol=1e-7)

    def test_fgmres_restart(self):
        """FGMRES should handle restart correctly."""
        A = fd3d(5, 5, 1)
        n = A.shape[0]
        x_true = np.random.randn(n)
        b = A @ x_true

        x, info = fgmres(A, b, restart=10, tol=1e-10, maxiter=100)

        assert info == 0
        np.testing.assert_allclose(x, x_true, rtol=1e-7)

    def test_fgmres_x0(self):
        """FGMRES should use initial guess."""
        A = tridiag(4, -1, -1, 10)
        x_true = np.random.randn(10)
        b = A @ x_true

        # Start close to solution
        x0 = x_true + 0.01 * np.random.randn(10)
        x, info = fgmres(A, b, x0=x0, tol=1e-10, maxiter=50)

        assert info == 0

    def test_fgmres_linear_operator(self):
        """FGMRES should work with LinearOperator."""
        A = fd3d(5, 5, 1)
        n = A.shape[0]

        def matvec(v):
            return A @ v

        A_op = LinearOperator((n, n), matvec=matvec)
        b = np.ones(n)

        x, info = fgmres(A_op, b, tol=1e-8)
        assert info == 0

    def test_fgmres_float32(self):
        """FGMRES should work with float32."""
        A = fd3d(5, 5, 1, dtype=np.float32)
        n = A.shape[0]
        b = np.ones(n, dtype=np.float32)

        x, info = fgmres(A, b, tol=1e-6)
        assert np.all(np.isfinite(x))


# =============================================================================
# FGMREZ Tests (Complex)
# =============================================================================

class TestFGMREZ:
    """Tests for FGMREZ (complex) solver."""

    def test_fgmrez_complex_system(self):
        """FGMREZ should solve complex system."""
        from scipy.sparse import eye as speye
        A = fd3d(5, 5, 1).astype(np.complex128)
        n = A.shape[0]
        A = A + 0.1j * speye(n, format='csr', dtype=np.complex128)

        x_true = np.random.randn(n) + 1j * np.random.randn(n)
        b = A @ x_true

        x, info = fgmrez(A, b, tol=1e-10, maxiter=100)

        assert info == 0
        np.testing.assert_allclose(x, x_true, rtol=1e-8)

    def test_fgmrez_with_preconditioner(self):
        """FGMREZ should work with complex ILU preconditioner."""
        from scipy.sparse import eye as speye
        A = fd3d(5, 5, 1).astype(np.complex128)
        n = A.shape[0]
        A = A + 0.1j * speye(n, format='csr', dtype=np.complex128)

        x_true = np.random.randn(n) + 1j * np.random.randn(n)
        b = A @ x_true

        M = ilu0(A)
        x, info = fgmrez(A, b, M=M, tol=1e-10, maxiter=100)

        assert info == 0
        np.testing.assert_allclose(x, x_true, rtol=1e-8)


# =============================================================================
# PCG Tests
# =============================================================================

class TestPCG:
    """Tests for Preconditioned Conjugate Gradient solver."""

    def test_pcg_converges(self):
        """PCG should converge for SPD system."""
        A = fd3d(5, 5, 1)  # SPD matrix
        n = A.shape[0]
        x_true = np.random.randn(n)
        b = A @ x_true

        x, info = pcg(A, b, tol=1e-10, maxiter=100)

        assert info == 0
        np.testing.assert_allclose(x, x_true, rtol=1e-8)

    def test_pcg_with_preconditioner(self):
        """PCG should work with ILU preconditioner."""
        A = fd3d(5, 5, 1)
        n = A.shape[0]
        x_true = np.random.randn(n)
        b = A @ x_true

        M = ilu0(A)
        x, info = pcg(A, b, M=M, tol=1e-10, maxiter=50)

        assert info == 0
        np.testing.assert_allclose(x, x_true, rtol=1e-7)

    def test_pcg_callable_preconditioner(self):
        """PCG should work with callable preconditioner."""
        A = fd3d(5, 5, 1)
        n = A.shape[0]
        x_true = np.random.randn(n)
        b = A @ x_true

        ilu = ilu0(A)
        x, info = pcg(A, b, M=lambda v: ilu.solve(v), tol=1e-10, maxiter=50)

        assert info == 0

    def test_pcg_x0(self):
        """PCG should use initial guess."""
        A = fd3d(5, 5, 1)
        n = A.shape[0]
        x_true = np.random.randn(n)
        b = A @ x_true

        x0 = x_true + 0.01 * np.random.randn(n)
        x, info = pcg(A, b, x0=x0, tol=1e-10, maxiter=50)

        assert info == 0

    def test_pcg_zero_rhs(self):
        """PCG with zero RHS should return zero."""
        A = fd3d(5, 5, 1)
        n = A.shape[0]
        b = np.zeros(n)

        x, info = pcg(A, b, tol=1e-10)

        assert info == 0
        np.testing.assert_allclose(x, np.zeros(n), atol=1e-12)

    def test_pcg_not_converged_returns_info(self):
        """PCG should return info > 0 when not converged."""
        A = fd3d(10, 10, 1)
        n = A.shape[0]
        b = np.ones(n)

        x, info = pcg(A, b, tol=1e-15, maxiter=2)

        assert info > 0

    def test_pcg_float32(self):
        """PCG should work with float32."""
        A = fd3d(5, 5, 1, dtype=np.float32)
        n = A.shape[0]
        b = np.ones(n, dtype=np.float32)

        x, info = pcg(A, b, tol=1e-5)
        assert np.all(np.isfinite(x))


# =============================================================================
# Planczos Tests
# =============================================================================

class TestPlanczos:
    """Tests for Preconditioned Lanczos eigenvalue estimation."""

    def test_planczos_returns_eigenvalues(self):
        """Planczos should return eigenvalue estimates."""
        A = fd3d(5, 5, 1)
        n = A.shape[0]
        v0 = np.ones(n)

        eigs = planczos(A, v0, m=10)

        assert len(eigs) > 0
        assert len(eigs) <= 10

    def test_planczos_eigenvalues_positive(self):
        """Planczos eigenvalues should be positive for SPD matrix."""
        A = fd3d(5, 5, 1)  # SPD matrix
        n = A.shape[0]
        v0 = np.ones(n)

        eigs = planczos(A, v0, m=10)

        assert np.all(eigs > 0)

    def test_planczos_extreme_eigenvalues(self):
        """Planczos should estimate extreme eigenvalues reasonably."""
        A = tridiag(4, -1, -1, 20)
        A_dense = A.toarray()
        true_eigs = np.linalg.eigvalsh(A_dense)
        v0 = np.ones(20)

        eigs = planczos(A, v0, m=30)

        # Lanczos should capture extreme eigenvalues
        assert np.min(eigs) < 1.5 * np.min(true_eigs)
        assert np.max(eigs) > 0.7 * np.max(true_eigs)

    def test_planczos_with_preconditioner(self):
        """Planczos should work with ILU preconditioner."""
        A = fd3d(5, 5, 1)
        n = A.shape[0]
        v0 = np.ones(n)
        M = ilu0(A)

        eigs = planczos(A, v0, M=M, m=10)

        assert len(eigs) > 0
        assert np.all(eigs > 0)

    def test_planczos_condition_number(self):
        """Planczos should estimate condition number."""
        A = fd3d(5, 5, 1)
        n = A.shape[0]
        v0 = np.ones(n)

        eigs = planczos(A, v0, m=20)
        cond_est = np.max(eigs) / np.min(eigs)

        # Condition number should be reasonable for FD matrix
        assert cond_est > 1
        assert cond_est < 1000

    def test_planczos_float32(self):
        """Planczos should work with float32."""
        A = fd3d(5, 5, 1, dtype=np.float32)
        n = A.shape[0]
        v0 = np.ones(n, dtype=np.float32)

        eigs = planczos(A, v0, m=10)

        assert len(eigs) > 0


# =============================================================================
# Integration Tests (Larger Problems)
# =============================================================================

class TestIntegration:
    """Integration tests with larger problems."""

    def test_fgmres_10x10x10(self):
        """FGMRES convergence on 10x10x10 3D problem."""
        np.random.seed(42)
        A = fd3d(10, 10, 10)
        n = A.shape[0]
        x_true = np.random.randn(n)
        b = A @ x_true

        ilu = ilu0(A)
        x, info = fgmres(A, b, M=ilu, tol=1e-10, maxiter=200)

        assert info == 0
        np.testing.assert_allclose(x, x_true, rtol=1e-7)

    def test_pcg_10x10x10(self):
        """PCG convergence on 10x10x10 3D problem."""
        np.random.seed(43)
        A = fd3d(10, 10, 10)
        n = A.shape[0]
        x_true = np.random.randn(n)
        b = A @ x_true

        ilu = ilu0(A)
        x, info = pcg(A, b, M=ilu, tol=1e-10, maxiter=200)

        assert info == 0
        np.testing.assert_allclose(x, x_true, rtol=1e-7)
