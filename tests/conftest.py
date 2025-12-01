"""
Pytest configuration and shared fixtures for schurilu tests.
"""

import numpy as np
import pytest
from scipy.sparse import diags

from schurilu.utils import fd3d

__all__ = ["fd3d", "tridiag"]


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)


def tridiag(a, b, c, n, dtype=np.float64):
    """
    Generates a tridiagonal matrix of size n x n.
    [b, a, c] <=> [-1, 0, 1] diagonals
    """
    return diags(
        [b * np.ones(n - 1, dtype=dtype),
         a * np.ones(n, dtype=dtype),
         c * np.ones(n - 1, dtype=dtype)],
        [-1, 0, 1], format="csr"
    )


@pytest.fixture
def small_2d_matrix():
    """5x5 2D finite difference matrix."""
    return fd3d(5, 5, 1)


@pytest.fixture
def medium_3d_matrix():
    """7x7x7 3D finite difference matrix (343 unknowns)."""
    return fd3d(7, 7, 7)


@pytest.fixture
def tridiagonal_matrix():
    """Simple 10x10 tridiagonal matrix."""
    return tridiag(4, -1, -1, 10)
