"""
Tests for utility functions (fd3d).
"""

import numpy as np
import pytest

from schurilu.utils import fd3d


# =============================================================================
# fd3d Structure Tests
# =============================================================================

class TestFD3DStructure:
    """Tests for fd3d matrix structure correctness."""

    def test_1d_laplacian_structure(self):
        """1D Laplacian should be tridiagonal: -1, 2, -1."""
        A = fd3d(5, 1, 1)
        A_dense = A.toarray()

        # Check dimensions
        assert A.shape == (5, 5)

        # Check diagonal = 2
        np.testing.assert_array_equal(np.diag(A_dense), [2, 2, 2, 2, 2])

        # Check off-diagonals = -1
        np.testing.assert_array_equal(np.diag(A_dense, 1), [-1, -1, -1, -1])
        np.testing.assert_array_equal(np.diag(A_dense, -1), [-1, -1, -1, -1])

        # Check sparsity (only 3 diagonals)
        assert A.nnz == 5 + 4 + 4  # diagonal + 2 off-diagonals

    def test_2d_laplacian_structure(self):
        """2D Laplacian should have 5-point stencil: diagonal=4, off-diag=-1."""
        nx, ny = 4, 4
        A = fd3d(nx, ny, 1)
        A_dense = A.toarray()
        n = nx * ny

        # Check dimensions
        assert A.shape == (n, n)

        # Check diagonal = 4
        np.testing.assert_array_equal(np.diag(A_dense), 4 * np.ones(n))

        # Check x-direction coupling (distance 1)
        for i in range(n - 1):
            if (i + 1) % nx != 0:  # not at x boundary
                assert A_dense[i, i + 1] == -1
                assert A_dense[i + 1, i] == -1

        # Check y-direction coupling (distance nx)
        for i in range(n - nx):
            assert A_dense[i, i + nx] == -1
            assert A_dense[i + nx, i] == -1

    def test_3d_laplacian_structure(self):
        """3D Laplacian should have 7-point stencil: diagonal=6, off-diag=-1."""
        nx, ny, nz = 3, 3, 3
        A = fd3d(nx, ny, nz)
        A_dense = A.toarray()
        n = nx * ny * nz

        # Check dimensions
        assert A.shape == (n, n)

        # Check diagonal = 6
        np.testing.assert_array_equal(np.diag(A_dense), 6 * np.ones(n))

        # Check interior point has 6 neighbors with -1
        # Interior point: (1,1,1) -> index = 1 + 1*nx + 1*nx*ny = 1 + 3 + 9 = 13
        interior_idx = 1 + 1 * nx + 1 * nx * ny
        row = A_dense[interior_idx, :]
        assert row[interior_idx] == 6  # diagonal
        assert row[interior_idx - 1] == -1  # x-
        assert row[interior_idx + 1] == -1  # x+
        assert row[interior_idx - nx] == -1  # y-
        assert row[interior_idx + nx] == -1  # y+
        assert row[interior_idx - nx * ny] == -1  # z-
        assert row[interior_idx + nx * ny] == -1  # z+
        # Only 7 nonzeros in this row
        assert np.count_nonzero(row) == 7

    def test_1d_eigenvalues(self):
        """1D Laplacian eigenvalues should match analytical formula."""
        n = 10
        A = fd3d(n, 1, 1)
        eigs = np.linalg.eigvalsh(A.toarray())

        # Analytical: lambda_k = 2 - 2*cos(k*pi/(n+1)), k=1,...,n
        expected = [2 - 2 * np.cos(k * np.pi / (n + 1)) for k in range(1, n + 1)]

        np.testing.assert_allclose(sorted(eigs), sorted(expected), rtol=1e-10)


# =============================================================================
# fd3d Properties Tests
# =============================================================================

class TestFD3DProperties:
    """Tests for fd3d matrix properties."""

    def test_symmetric(self):
        """fd3d should produce symmetric matrices."""
        for args in [(5, 1, 1), (5, 5, 1), (4, 4, 4)]:
            A = fd3d(*args)
            diff = A - A.T
            assert diff.nnz == 0 or np.abs(diff.data).max() < 1e-14

    def test_positive_definite(self):
        """fd3d without shift should be SPD (all eigenvalues > 0)."""
        A = fd3d(5, 5, 5)
        eigs = np.linalg.eigvalsh(A.toarray())
        assert np.all(eigs > 0)

    def test_shift_reduces_eigenvalues(self):
        """Diagonal shift should reduce eigenvalues."""
        A0 = fd3d(5, 5, 1, shift=0.0)
        A1 = fd3d(5, 5, 1, shift=1.0)

        eigs0 = np.linalg.eigvalsh(A0.toarray())
        eigs1 = np.linalg.eigvalsh(A1.toarray())

        # shift=1.0 subtracts 1 from all eigenvalues
        np.testing.assert_allclose(eigs1, eigs0 - 1.0, rtol=1e-10)

    def test_dtype_float32(self):
        """fd3d should respect dtype argument."""
        A = fd3d(5, 5, 1, dtype=np.float32)
        assert A.dtype == np.float32

    def test_dtype_float64(self):
        """fd3d should default to float64."""
        A = fd3d(5, 5, 1)
        assert A.dtype == np.float64

    def test_csr_format(self):
        """fd3d should return CSR matrix."""
        A = fd3d(5, 5, 5)
        assert A.format == 'csr'


# =============================================================================
# fd3d Edge Cases
# =============================================================================

class TestFD3DEdgeCases:
    """Tests for fd3d edge cases."""

    def test_1x1x1(self):
        """1x1x1 grid should be a 1x1 matrix with value 0."""
        A = fd3d(1, 1, 1)
        assert A.shape == (1, 1)
        # 1D Laplacian of size 1: diagonal = 2, no off-diagonal neighbors
        assert A[0, 0] == 2

    def test_ny1_nz_gt_1_raises(self):
        """ny=1 with nz>1 should raise ValueError."""
        with pytest.raises(ValueError, match="ny must be > 1"):
            fd3d(5, 1, 5)

    def test_large_shift_indefinite(self):
        """Large shift should make matrix indefinite."""
        A = fd3d(5, 5, 1, shift=10.0)  # diagonal 4 - 10 = -6
        eigs = np.linalg.eigvalsh(A.toarray())
        assert np.min(eigs) < 0  # has negative eigenvalues

