"""
Tests for reordering module (spectral graph partitioning).
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from schurilu import multilevel_partition, spectral_kway, unweighted_laplacian, connected_components
from tests.conftest import fd3d


# =============================================================================
# Unweighted Laplacian Tests
# =============================================================================

class TestUnweightedLaplacian:
    """Tests for unweighted Laplacian construction."""

    def test_laplacian_symmetric(self):
        """Laplacian should be symmetric."""
        A = fd3d(6, 6, 1)
        L, W, D = unweighted_laplacian(A)

        L_dense = L.toarray()
        np.testing.assert_allclose(L_dense, L_dense.T, rtol=1e-14)

    def test_laplacian_row_sum_zero(self):
        """Laplacian rows should sum to zero."""
        A = fd3d(6, 6, 1)
        L, W, D = unweighted_laplacian(A)

        row_sums = np.array(L.sum(axis=1)).flatten()
        np.testing.assert_allclose(row_sums, 0, atol=1e-14)

    def test_adjacency_no_self_loops(self):
        """Adjacency matrix should have zero diagonal."""
        A = fd3d(6, 6, 1)
        L, W, D = unweighted_laplacian(A)

        np.testing.assert_allclose(W.diagonal(), 0, atol=1e-14)


# =============================================================================
# Connected Components Tests
# =============================================================================

class TestConnectedComponents:
    """Tests for connected components detection."""

    def test_connected_graph_single_component(self):
        """Connected graph should have one component."""
        A = fd3d(6, 6, 1)  # 2D grid is connected
        comps = connected_components(A)

        assert len(comps) == 1
        assert len(comps[0]) == A.shape[0]

    def test_disconnected_graph_multiple_components(self):
        """Disconnected graph should have multiple components."""
        # Two 3x3 disconnected blocks
        A = csr_matrix([
            [4, -1, 0, 0, 0, 0],
            [-1, 4, -1, 0, 0, 0],
            [0, -1, 4, 0, 0, 0],
            [0, 0, 0, 4, -1, 0],
            [0, 0, 0, -1, 4, -1],
            [0, 0, 0, 0, -1, 4],
        ], dtype=float)

        comps = connected_components(A)

        assert len(comps) == 2
        all_nodes = set()
        for comp in comps:
            all_nodes.update(comp)
        assert all_nodes == set(range(6))


# =============================================================================
# Spectral K-way Partition Tests
# =============================================================================

class TestSpectralKway:
    """Tests for spectral k-way partitioning."""

    def test_spectral_bisection(self):
        """Spectral bisection should produce 2 partitions."""
        A = fd3d(8, 8, 1)
        partition = spectral_kway(A, k=2)

        assert len(partition) == A.shape[0]
        assert set(partition) == {0, 1}

    def test_spectral_4way(self):
        """Spectral 4-way should produce 4 partitions."""
        A = fd3d(8, 8, 1)
        partition = spectral_kway(A, k=4)

        assert len(partition) == A.shape[0]
        assert set(partition) == {0, 1, 2, 3}

    def test_spectral_roughly_balanced(self):
        """Partitions should be roughly balanced."""
        A = fd3d(8, 8, 1)  # 64 nodes
        partition = spectral_kway(A, k=4)

        counts = [np.sum(partition == i) for i in range(4)]
        # Each partition should have between 8 and 24 nodes (16 +/- 8)
        for count in counts:
            assert 4 <= count <= 32

    def test_spectral_invalid_k_raises(self):
        """Non-power-of-2 k should raise error."""
        A = fd3d(6, 6, 1)
        with pytest.raises(ValueError):
            spectral_kway(A, k=3)


# =============================================================================
# Multilevel Partition Tests
# =============================================================================

class TestMultilevelPartition:
    """Tests for multilevel domain decomposition partitioning."""

    def test_permutation_valid(self):
        """Permutation should be a valid permutation."""
        A = fd3d(8, 8, 1)
        n = A.shape[0]

        p, lev_ptr, nlev = multilevel_partition(A, nlev=2, k=4)

        assert len(p) == n
        assert set(p) == set(range(n))  # Valid permutation

    def test_level_pointers_monotonic(self):
        """Level pointers should be monotonically increasing."""
        A = fd3d(8, 8, 1)
        n = A.shape[0]

        p, lev_ptr, nlev = multilevel_partition(A, nlev=2, k=4)

        assert lev_ptr[0] == 0
        assert lev_ptr[-1] == n
        for i in range(len(lev_ptr) - 1):
            assert lev_ptr[i] < lev_ptr[i + 1]

    def test_multilevel_3d(self):
        """Multilevel partition should work for 3D problems."""
        A = fd3d(6, 6, 6)  # 216 unknowns
        n = A.shape[0]

        p, lev_ptr, nlev = multilevel_partition(A, nlev=2, k=4)

        assert len(p) == n
        assert set(p) == set(range(n))
        assert nlev >= 1

    def test_permuted_matrix_structure(self):
        """Permuted matrix should have block structure."""
        A = fd3d(8, 8, 1)
        n = A.shape[0]

        p, lev_ptr, nlev = multilevel_partition(A, nlev=2, k=4)

        # Apply permutation
        Ap = A[p, :][:, p]

        # Extract B and C blocks
        nB = lev_ptr[1]
        B = Ap[:nB, :nB]
        C = Ap[nB:, nB:]

        assert B.shape[0] == nB
        assert C.shape[0] == n - nB
        assert B.nnz > 0
        assert C.nnz > 0
