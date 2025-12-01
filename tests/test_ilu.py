"""
Tests for ILU factorization modules (ilu0, iluk, ilut).
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, coo_matrix, diags

from schurilu import ilu0, iluk, ilut, ILUResult
from tests.conftest import fd3d, tridiag


# =============================================================================
# ILU0 Tests
# =============================================================================


class TestILU0:
    """Tests for ILU0 factorization."""

    def test_ilu0_returns_ilu_result(self):
        """ILU0 should return an ILUResult object."""
        A = fd3d(5, 5, 1)
        result = ilu0(A)
        assert isinstance(result, ILUResult)

    def test_ilu0_dimensions(self):
        """ILU0 should return factors with correct dimensions."""
        A = fd3d(5, 5, 1)
        n = A.shape[0]
        result = ilu0(A)

        assert result.L.shape == (n, n)
        assert result.U.shape == (n, n)
        assert result.D.shape == (n,)

    def test_ilu0_tridiagonal_exact(self):
        """ILU0 on tridiagonal matrix should give exact LU."""
        A = tridiag(4, -1, -1, 10)
        result = ilu0(A)

        L_complete, U_complete = result.to_complete()
        LU = L_complete @ U_complete

        np.testing.assert_allclose(LU.toarray(), A.toarray(), rtol=1e-12)

    def test_ilu0_solve(self):
        """ILU0 solve should work as preconditioner."""
        A = fd3d(5, 5, 1)
        n = A.shape[0]
        x_true = np.random.randn(n)
        b = A @ x_true

        result = ilu0(A)
        x_solved = result.solve(b)

        # ILU0 won't be exact, but should reduce residual
        residual = np.linalg.norm(A @ x_solved - b) / np.linalg.norm(b)
        assert residual < 0.5

    def test_ilu0_schur_complement(self):
        """ILU0 with nB < n should return Schur complement."""
        A = fd3d(5, 5, 1)
        n = A.shape[0]
        nB = n // 2

        result = ilu0(A, nB=nB)

        assert result.nB == nB
        assert result.E is not None
        assert result.F is not None
        assert result.S is not None
        assert result.S.shape == (n - nB, n - nB)

    def test_ilu0_float32(self):
        """ILU0 should work with float32."""
        A = fd3d(5, 5, 1, dtype=np.float32)
        result = ilu0(A)
        assert result.dtype == np.float32

    def test_ilu0_complex128(self):
        """ILU0 should work with complex128."""
        A = fd3d(5, 5, 1)
        A = A.astype(np.complex128) + 0.1j * diags(
            [1], [0], shape=A.shape, format="csr"
        )
        result = ilu0(A)
        assert result.dtype == np.complex128

    def test_ilu0_linear_operator(self):
        """ILU0 should provide a linear operator."""
        A = fd3d(5, 5, 1)
        result = ilu0(A)
        M = result.to_linear_operator()

        b = np.ones(A.shape[0])
        x = M @ b
        assert x.shape == b.shape


# =============================================================================
# ILUK Tests
# =============================================================================


class TestILUK:
    """Tests for ILUK factorization."""

    def test_iluk_returns_ilu_result(self):
        """ILUK should return an ILUResult object."""
        A = fd3d(5, 5, 1)
        result = iluk(A, lfil=1)
        assert isinstance(result, ILUResult)

    def test_iluk_tridiagonal_exact(self):
        """ILUK on tridiagonal matrix should give exact LU."""
        A = tridiag(4, -1, -1, 10)
        result = iluk(A, lfil=1)

        L_complete, U_complete = result.to_complete()
        LU = L_complete @ U_complete

        np.testing.assert_allclose(LU.toarray(), A.toarray(), rtol=1e-12)

    def test_iluk_high_lfil_exact(self):
        """ILUK with high lfil should give exact LU."""
        A = fd3d(5, 5, 1)
        n = A.shape[0]
        result = iluk(A, lfil=n)

        L_complete, U_complete = result.to_complete()
        LU = L_complete @ U_complete

        np.testing.assert_allclose(LU.toarray(), A.toarray(), rtol=1e-10, atol=1e-12)

    def test_iluk_schur_complement(self):
        """ILUK with nB < n should return Schur complement."""
        A = fd3d(5, 5, 1)
        n = A.shape[0]
        nB = n // 2

        result = iluk(A, lfil=3, nB=nB)

        assert result.nB == nB
        assert result.S is not None
        assert result.S.shape == (n - nB, n - nB)

    def test_iluk_high_lfil_schur_exact(self):
        """ILUK with high lfil should give exact Schur complement."""
        A = fd3d(5, 5, 1)
        A_dense = A.toarray()
        n = A.shape[0]
        nB = n // 2

        # Compute exact Schur complement
        B = A_dense[:nB, :nB]
        F_block = A_dense[:nB, nB:]
        E_block = A_dense[nB:, :nB]
        C = A_dense[nB:, nB:]
        S_exact = C - E_block @ np.linalg.inv(B) @ F_block

        result = iluk(A, lfil=n, nB=nB)

        np.testing.assert_allclose(result.S.toarray(), S_exact, rtol=1e-10)

    def test_iluk_solve(self):
        """ILUK solve should give exact solution for high lfil."""
        A = tridiag(4, -1, -1, 10)
        x_true = np.random.randn(10)
        b = A @ x_true

        result = iluk(A, lfil=10)
        x_solved = result.solve(b)

        np.testing.assert_allclose(x_solved, x_true, rtol=1e-10)

    def test_iluk_complex128(self):
        """ILUK should work with complex128."""
        A = fd3d(5, 5, 1)
        A = A.astype(np.complex128) + 0.1j * diags(
            [1], [0], shape=A.shape, format="csr"
        )
        result = iluk(A, lfil=3)
        assert result.dtype == np.complex128


# =============================================================================
# ILUT Tests
# =============================================================================


class TestILUT:
    """Tests for ILUT factorization."""

    def test_ilut_returns_ilu_result(self):
        """ILUT should return an ILUResult object."""
        A = fd3d(5, 5, 1)
        result = ilut(A, droptol=0.01, lfil=10)
        assert isinstance(result, ILUResult)

    def test_ilut_tridiagonal_exact(self):
        """ILUT on tridiagonal matrix should give exact LU."""
        A = tridiag(4, -1, -1, 10)
        result = ilut(A, droptol=0, lfil=10)

        L_complete, U_complete = result.to_complete()
        LU = L_complete @ U_complete

        np.testing.assert_allclose(LU.toarray(), A.toarray(), rtol=1e-12)

    def test_ilut_zero_droptol_exact(self):
        """ILUT with droptol=0 and high lfil should give exact LU."""
        A = fd3d(5, 5, 1)
        n = A.shape[0]
        result = ilut(A, droptol=0, lfil=n)

        L_complete, U_complete = result.to_complete()
        LU = L_complete @ U_complete

        np.testing.assert_allclose(LU.toarray(), A.toarray(), rtol=1e-10, atol=1e-12)

    def test_ilut_high_droptol_sparse(self):
        """ILUT with high droptol should produce sparser factors."""
        A = fd3d(5, 5, 1)

        result_sparse = ilut(A, droptol=0.5, lfil=100)
        result_dense = ilut(A, droptol=0, lfil=100)

        assert result_sparse.nnz <= result_dense.nnz

    def test_ilut_schur_complement(self):
        """ILUT with nB < n should return Schur complement."""
        A = fd3d(5, 5, 1)
        n = A.shape[0]
        nB = n // 2

        result = ilut(A, droptol=0.01, lfil=10, nB=nB)

        assert result.nB == nB
        assert result.S is not None

    def test_ilut_zero_droptol_schur_exact(self):
        """ILUT with droptol=0 should give exact Schur complement."""
        A = fd3d(5, 5, 1)
        A_dense = A.toarray()
        n = A.shape[0]
        nB = n // 2

        B = A_dense[:nB, :nB]
        F_block = A_dense[:nB, nB:]
        E_block = A_dense[nB:, :nB]
        C = A_dense[nB:, nB:]
        S_exact = C - E_block @ np.linalg.inv(B) @ F_block

        result = ilut(A, droptol=0, lfil=n, nB=nB)

        np.testing.assert_allclose(result.S.toarray(), S_exact, rtol=1e-10)

    def test_ilut_solve(self):
        """ILUT solve should give exact solution with droptol=0."""
        A = tridiag(4, -1, -1, 10)
        x_true = np.random.randn(10)
        b = A @ x_true

        result = ilut(A, droptol=0, lfil=10)
        x_solved = result.solve(b)

        np.testing.assert_allclose(x_solved, x_true, rtol=1e-10)

    def test_ilut_complex128(self):
        """ILUT should work with complex128."""
        A = fd3d(5, 5, 1)
        A = A.astype(np.complex128) + 0.1j * diags(
            [1], [0], shape=A.shape, format="csr"
        )
        result = ilut(A, droptol=0.01, lfil=10)
        assert result.dtype == np.complex128


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_ilu0_non_square_raises(self):
        """Non-square matrix should raise ValueError."""
        A = csr_matrix(np.random.randn(10, 5))
        with pytest.raises(ValueError, match="square"):
            ilu0(A)

    def test_iluk_non_square_raises(self):
        """Non-square matrix should raise ValueError."""
        A = csr_matrix(np.random.randn(10, 5))
        with pytest.raises(ValueError, match="square"):
            iluk(A)

    def test_ilut_non_square_raises(self):
        """Non-square matrix should raise ValueError."""
        A = csr_matrix(np.random.randn(10, 5))
        with pytest.raises(ValueError, match="square"):
            ilut(A)

    def test_ilu0_invalid_nB(self):
        """Invalid nB should raise ValueError."""
        A = fd3d(5, 5, 1)
        n = A.shape[0]

        with pytest.raises(ValueError, match="nB"):
            ilu0(A, nB=-1)
        with pytest.raises(ValueError, match="nB"):
            ilu0(A, nB=n + 1)

    def test_ilu0_1x1_matrix(self):
        """1x1 matrix should work."""
        A = csr_matrix([[5.0]])
        result = ilu0(A)
        assert result.n == 1
        assert result.D[0] == pytest.approx(1.0 / 5.0)

    def test_ilu0_coo_input(self):
        """COO format should be converted to CSR."""
        A_csr = fd3d(5, 5, 1)
        A_coo = coo_matrix(A_csr)
        result = ilu0(A_coo)
        assert result.L.shape == A_csr.shape

    def test_ilu0_integer_input(self):
        """Integer matrix should be converted to float."""
        A = csr_matrix([[4, -1, 0], [-1, 4, -1], [0, -1, 4]], dtype=np.int64)
        result = ilu0(A)
        assert result.dtype in [np.float64, np.float32]


# =============================================================================
# Numerical Robustness Tests
# =============================================================================


class TestNumericalRobustness:
    """Tests for numerical robustness."""

    def test_ilu0_zero_diagonal_handling(self):
        """Zero diagonal should be replaced with small value."""
        A = fd3d(5, 5, 1).toarray()
        A[5, 5] = 0.0
        A = csr_matrix(A)

        result = ilu0(A)
        assert np.all(np.abs(result.D) > 1e-15)

    def test_iluk_zero_diagonal_handling(self):
        """Zero diagonal should be replaced with small value."""
        A = fd3d(5, 5, 1).toarray()
        A[5, 5] = 0.0
        A = csr_matrix(A)

        result = iluk(A, lfil=5)
        assert np.all(np.abs(result.D) > 1e-15)

    def test_ilut_zero_diagonal_handling(self):
        """Zero diagonal should be replaced with small value."""
        A = fd3d(5, 5, 1).toarray()
        A[5, 5] = 0.0
        A = csr_matrix(A)

        result = ilut(A, droptol=0.01, lfil=10)
        assert np.all(np.abs(result.D) > 1e-15)


# =============================================================================
# Larger Problem Tests
# =============================================================================


class TestLargerProblems:
    """Tests with larger matrices."""

    def test_iluk_7x7x7_exact(self):
        """ILUK with high lfil gives exact LU on 7x7x7 problem."""
        A = fd3d(7, 7, 7)
        n = A.shape[0]
        assert n == 343

        result = iluk(A, lfil=n)
        L_complete, U_complete = result.to_complete()
        LU = L_complete @ U_complete

        np.testing.assert_allclose(LU.toarray(), A.toarray(), rtol=1e-10, atol=1e-12)

    def test_ilut_7x7x7_exact(self):
        """ILUT with droptol=0 gives exact LU on 7x7x7 problem."""
        A = fd3d(7, 7, 7)
        n = A.shape[0]

        result = ilut(A, droptol=0, lfil=n)
        L_complete, U_complete = result.to_complete()
        LU = L_complete @ U_complete

        np.testing.assert_allclose(LU.toarray(), A.toarray(), rtol=1e-10, atol=1e-12)

    def test_iluk_7x7x7_schur_exact(self):
        """ILUK Schur complement is exact on 7x7x7 problem."""
        A = fd3d(7, 7, 7)
        A_dense = A.toarray()
        n = A.shape[0]
        nB = 200

        B = A_dense[:nB, :nB]
        F_block = A_dense[:nB, nB:]
        E_block = A_dense[nB:, :nB]
        C = A_dense[nB:, nB:]
        S_exact = C - E_block @ np.linalg.inv(B) @ F_block

        result = iluk(A, lfil=n, nB=nB)

        np.testing.assert_allclose(result.S.toarray(), S_exact, rtol=1e-8, atol=1e-10)


# =============================================================================
# Modified ILU (MILU) Tests
# =============================================================================


class TestModifiedILU:
    """Tests for Modified ILU (MILU) functionality."""

    def test_ilu0_modified_row_sum(self):
        """Modified ILU should preserve row sums."""
        from scipy.sparse import eye as speye

        A = fd3d(5, 5, 5)  # 125 nodes
        n = A.shape[0]

        result = ilu0(A, modified=True)

        # Reconstruct LU
        L = result.L + speye(n, format="csr")
        D_diag = diags(1.0 / result.D, 0, format="csr")
        U = result.U + D_diag
        LU = L @ U

        # Verify row sums
        A_row_sum = np.array(A.sum(axis=1)).flatten()
        LU_row_sum = np.array(LU.sum(axis=1)).flatten()

        np.testing.assert_allclose(A_row_sum, LU_row_sum, rtol=1e-10, atol=1e-12)

    def test_iluk_modified_row_sum(self):
        """Modified ILUK should preserve row sums for high lfil."""
        from scipy.sparse import eye as speye

        A = fd3d(5, 5, 1)
        n = A.shape[0]

        result = iluk(A, lfil=n, modified=True)

        L = result.L + speye(n, format="csr")
        D_diag = diags(1.0 / result.D, 0, format="csr")
        U = result.U + D_diag
        LU = L @ U

        A_row_sum = np.array(A.sum(axis=1)).flatten()
        LU_row_sum = np.array(LU.sum(axis=1)).flatten()

        np.testing.assert_allclose(A_row_sum, LU_row_sum, rtol=1e-10, atol=1e-12)

    def test_ilut_modified_row_sum(self):
        """Modified ILUT should preserve row sums for zero droptol."""
        from scipy.sparse import eye as speye

        A = fd3d(5, 5, 1)
        n = A.shape[0]

        result = ilut(A, droptol=0, lfil=n, modified=True)

        L = result.L + speye(n, format="csr")
        D_diag = diags(1.0 / result.D, 0, format="csr")
        U = result.U + D_diag
        LU = L @ U

        A_row_sum = np.array(A.sum(axis=1)).flatten()
        LU_row_sum = np.array(LU.sum(axis=1)).flatten()

        np.testing.assert_allclose(A_row_sum, LU_row_sum, rtol=1e-10, atol=1e-12)

    def test_ilu0_modified_convergence_poisson(self):
        """Modified ILU should converge on Poisson problem."""
        from schurilu.krylov import fgmres

        A = fd3d(8, 8, 8)  # 512 nodes
        n = A.shape[0]
        b = np.ones(n)

        # Standard ILU0
        M0 = ilu0(A, modified=False)
        iter_count_std = [0]

        def callback_std(r):
            iter_count_std[0] += 1

        x0, info0 = fgmres(A, b, M=M0, tol=1e-8, maxiter=200, callback=callback_std)

        # Modified ILU0
        M1 = ilu0(A, modified=True)
        iter_count_mod = [0]

        def callback_mod(r):
            iter_count_mod[0] += 1

        x1, info1 = fgmres(A, b, M=M1, tol=1e-8, maxiter=200, callback=callback_mod)

        # Both should converge
        assert info0 == 0, f"Standard ILU0 failed to converge"
        assert info1 == 0, f"Modified ILU0 failed to converge"

        # Modified often converges faster for Poisson
        # At minimum, it should not be significantly worse
        assert iter_count_mod[0] <= iter_count_std[0] + 5


# =============================================================================
# List-Form Parameter Tests
# =============================================================================


class TestListFormParameters:
    """Tests for list-form parameters in ILUK and ILUT."""

    def test_iluk_list_lfil(self):
        """ILUK with list-form lfil should work."""
        A = fd3d(6, 6, 1)  # 36 nodes
        nB = A.shape[0] // 2

        # Use list form: [lfilb, lfilef, lfils]
        result = iluk(A, lfil=[2, 1, 3], nB=nB)

        assert result.L.shape == A.shape
        assert result.S is not None

        # Verify preconditioner works
        b = np.random.randn(A.shape[0])
        x = result.solve(b)
        assert np.isfinite(x).all()

    def test_iluk_list_lfil_two_elements(self):
        """ILUK with two-element lfil list should work."""
        A = fd3d(6, 6, 1)
        nB = A.shape[0] // 2

        # Two-element form: [lfilb, lfilef]
        result = iluk(A, lfil=[2, 1], nB=nB)

        assert result.S is not None
        b = np.random.randn(A.shape[0])
        x = result.solve(b)
        assert np.isfinite(x).all()

    def test_ilut_list_droptol(self):
        """ILUT with list-form droptol should work."""
        A = fd3d(6, 6, 1)
        nB = A.shape[0] // 2

        # List form droptol: [drolb, drolef, drols]
        result = ilut(A, droptol=[1e-3, 1e-4, 1e-3], lfil=10, nB=nB)

        assert result.S is not None
        b = np.random.randn(A.shape[0])
        x = result.solve(b)
        assert np.isfinite(x).all()

    def test_ilut_list_lfil(self):
        """ILUT with list-form lfil should work."""
        A = fd3d(6, 6, 1)
        nB = A.shape[0] // 2

        # List form lfil: [lfilb, lfilef, lfils]
        result = ilut(A, droptol=1e-3, lfil=[10, 5, 15], nB=nB)

        assert result.S is not None

    def test_ilut_list_both_params(self):
        """ILUT with both list-form droptol and lfil should work."""
        A = fd3d(6, 6, 1)
        nB = A.shape[0] // 2

        result = ilut(A, droptol=[1e-3, 1e-4, 1e-3], lfil=[10, 5, 15], nB=nB)

        assert result.S is not None
        b = np.random.randn(A.shape[0])
        x = result.solve(b)
        assert np.isfinite(x).all()


# =============================================================================
# Additional Complex Type Tests
# =============================================================================


class TestComplexTypes:
    """Additional tests for complex data types."""

    def test_iluk_complex_solve(self):
        """ILUK should solve complex systems."""
        A = fd3d(5, 5, 1).astype(np.complex128)
        A = A + 0.1j * diags([1], [0], shape=A.shape, format="csr")
        A = A + 10 * diags([1], [0], shape=A.shape, format="csr")  # Diagonal dominant

        result = iluk(A, lfil=3)

        assert result.L.dtype == np.complex128
        assert result.D.dtype == np.complex128

        b = np.random.randn(A.shape[0]) + 1j * np.random.randn(A.shape[0])
        x = result.solve(b)

        assert np.isfinite(x).all()
        # Check residual
        residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        assert residual < 1.0

    def test_ilut_complex_solve(self):
        """ILUT should solve complex systems."""
        A = fd3d(5, 5, 1).astype(np.complex128)
        A = A + 0.1j * diags([1], [0], shape=A.shape, format="csr")
        A = A + 10 * diags([1], [0], shape=A.shape, format="csr")

        result = ilut(A, droptol=1e-3, lfil=10)

        assert result.L.dtype == np.complex128

        b = np.random.randn(A.shape[0]) + 1j * np.random.randn(A.shape[0])
        x = result.solve(b)

        assert np.isfinite(x).all()
        residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        assert residual < 1.0

    def test_iluk_complex_schur(self):
        """ILUK Schur complement should work with complex matrices."""
        A = fd3d(5, 5, 1).astype(np.complex128)
        A = A + 0.1j * diags([1], [0], shape=A.shape, format="csr")
        nB = A.shape[0] // 2

        result = iluk(A, lfil=3, nB=nB)

        assert result.S is not None
        assert result.S.dtype == np.complex128
        assert result.E.dtype == np.complex128
        assert result.F.dtype == np.complex128
