"""
Tests for rotated 2D Laplacian (rlap2d).
"""

import numpy as np
import pytest

from schurilu.utils import fd3d


def _get_rlap2d():
    # Local import to avoid import error before implementation
    from schurilu.utils import rlap2d
    return rlap2d


class TestRLap2DStructure:
    def test_reduces_to_5pt_when_epsilon_one(self):
        nx, ny = 4, 5
        rlap2d = _get_rlap2d()
        A = rlap2d(nx, ny, epsilon=1.0, theta=0.37)
        B = fd3d(nx, ny, 1)
        np.testing.assert_array_equal(A.toarray(), B.toarray())

    def test_theta_zero_axis_aligned_coeffs(self):
        nx, ny = 4, 4
        eps = 3.0
        rlap2d = _get_rlap2d()
        A = rlap2d(nx, ny, epsilon=eps, theta=0.0)
        M = A.toarray()
        n = nx * ny
        # Axis-aligned (theta=0): N,S = -epsilon; E,W = -1; diag = 2*(epsilon+1)
        diag = np.diag(M)
        np.testing.assert_array_equal(diag, 2 * (eps + 1) * np.ones(n))
        for j in range(ny):
            for i in range(nx):
                r = j * nx + i
                if i > 0:
                    assert M[r, r - 1] == -1
                if i < nx - 1:
                    assert M[r, r + 1] == -1
                if j > 0:
                    assert M[r, r - nx] == -eps
                if j < ny - 1:
                    assert M[r, r + nx] == -eps
        # No corner couplings at theta=0 (check around an interior point)
        if nx >= 3 and ny >= 3:
            r = 1 + 1 * nx
            assert M[r, r - nx - 1] == 0
            assert M[r, r - nx + 1] == 0
            assert M[r, r + nx - 1] == 0
            assert M[r, r + nx + 1] == 0

    def test_corners_nonzero_for_rotated_anisotropy(self):
        nx, ny = 5, 5
        rlap2d = _get_rlap2d()
        A = rlap2d(nx, ny, epsilon=2.0, theta=np.pi / 4)
        M = A.toarray()
        r = 2 + 2 * nx
        # Interior row should have 9 nonzeros
        assert np.count_nonzero(M[r, :]) == 9
        # Corner couplings nonzero
        assert M[r, r - nx - 1] != 0
        assert M[r, r - nx + 1] != 0
        assert M[r, r + nx - 1] != 0
        assert M[r, r + nx + 1] != 0


class TestRLap2DProperties:
    def test_symmetric_spd(self):
        rlap2d = _get_rlap2d()
        A = rlap2d(4, 4, epsilon=2.5, theta=0.7)
        M = A.toarray()
        np.testing.assert_allclose(M, M.T, rtol=0, atol=1e-14)
        w = np.linalg.eigvalsh(M)
        assert np.all(w > 0)

    def test_shift_reduces_eigenvalues(self):
        rlap2d = _get_rlap2d()
        A0 = rlap2d(5, 6, epsilon=1.7, theta=0.3, shift=0.0)
        A1 = rlap2d(5, 6, epsilon=1.7, theta=0.3, shift=0.5)
        e0 = np.linalg.eigvalsh(A0.toarray())
        e1 = np.linalg.eigvalsh(A1.toarray())
        np.testing.assert_allclose(e1, e0 - 0.5, rtol=1e-10)

    def test_dtype_respected(self):
        rlap2d = _get_rlap2d()
        A = rlap2d(3, 4, epsilon=2.0, theta=0.2, dtype=np.float32)
        assert A.dtype == np.float32
