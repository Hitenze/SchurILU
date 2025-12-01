"""
Spectral graph partitioning for multilevel domain decomposition.

This module provides Fiedler vector-based spectral partitioning for
creating multilevel orderings suitable for GeMSLR preconditioners.
"""

from typing import List, Tuple, Optional, Union, Any
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, diags, issparse, spmatrix
from scipy.sparse.linalg import eigsh
from collections import deque


def unweighted_laplacian(
    A: Union[spmatrix, npt.ArrayLike]
) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
    """
    Create unweighted graph Laplacian from sparse matrix.

    Constructs L = D - W where W is the unweighted adjacency matrix
    derived from |A| + |A^T| with diagonal removed.

    Parameters
    ----------
    A : sparse matrix
        Input matrix (pattern used to define graph).

    Returns
    -------
    L : csr_matrix
        Graph Laplacian matrix.
    W : csr_matrix
        Unweighted adjacency matrix (symmetric, no self-loops).
    D : csr_matrix
        Degree matrix (diagonal).
    """
    n = A.shape[0]

    # Make symmetric and take absolute values
    A_abs = abs(A)
    if issparse(A_abs):
        A_sym = A_abs + A_abs.T
    else:
        A_sym = A_abs + A_abs.T

    # Convert to CSR if needed
    if not isinstance(A_sym, csr_matrix):
        A_sym = csr_matrix(A_sym)

    # Remove diagonal (self-loops)
    A_sym = A_sym.tolil()
    A_sym.setdiag(0)
    A_sym = A_sym.tocsr()

    # Create unweighted adjacency (all nonzeros become 1)
    W = A_sym.copy()
    W.data = np.ones_like(W.data)

    # Degree matrix
    degrees = np.array(W.sum(axis=1)).flatten()
    D = diags(degrees, 0, format="csr")

    # Laplacian
    L = D - W

    return L, W, D


def connected_components(A: Union[spmatrix, npt.ArrayLike]) -> List[List[int]]:
    """
    Find connected components in the graph defined by matrix A.

    Parameters
    ----------
    A : sparse matrix
        Input matrix (pattern defines graph connectivity).

    Returns
    -------
    components : list of lists
        Each element is a list of node indices in that component.
    """
    n = A.shape[0]

    if not issparse(A):
        A = csr_matrix(A)

    # Make symmetric for undirected graph
    A_sym = A + A.T
    A_csr = A_sym.tocsr()

    marker = np.zeros(n, dtype=np.int32)
    components = []

    for start in range(n):
        if marker[start] == 0:
            # BFS from this node
            component = []
            queue = deque([start])
            marker[start] = 1

            while queue:
                node = queue.popleft()
                component.append(node)

                # Get neighbors
                row_start = A_csr.indptr[node]
                row_end = A_csr.indptr[node + 1]
                neighbors = A_csr.indices[row_start:row_end]

                for neighbor in neighbors:
                    if marker[neighbor] == 0:
                        marker[neighbor] = 1
                        queue.append(neighbor)

            components.append(component)

    return components


def _fiedler_vector(L: spmatrix, n: int) -> npt.NDArray[Any]:
    """
    Compute Fiedler vector (eigenvector of 2nd smallest eigenvalue).

    Parameters
    ----------
    L : sparse matrix
        Graph Laplacian.
    n : int
        Size of matrix.

    Returns
    -------
    v : ndarray
        Fiedler vector.
    """
    if n <= 1:
        return np.ones(n)

    if n <= 32:
        # Small matrix: use dense eigenvalue solver
        L_dense = L.toarray() if issparse(L) else L
        eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
        # Second smallest eigenvalue (first is ~0 for connected graph)
        idx = np.argsort(eigenvalues)[1]
        v = eigenvectors[:, idx]
    else:
        # Large matrix: use sparse solver
        try:
            # Find 2 smallest eigenvalues
            k = min(2, n - 1)
            ncv = min(max(2 * k + 1, 20), n)
            eigenvalues, eigenvectors = eigsh(L, k=k, which="SM", ncv=ncv)
            # Get eigenvector for 2nd smallest
            idx = np.argsort(eigenvalues)
            if len(idx) >= 2:
                v = eigenvectors[:, idx[1]]
            else:
                v = eigenvectors[:, idx[0]]
        except Exception:
            # Fallback to random vector
            v = np.random.randn(n)

    return v


def _spectral_bisection(
    W: spmatrix, nodes: Optional[npt.ArrayLike] = None
) -> Tuple[List[int], List[int]]:
    """
    Perform spectral bisection on a graph.

    Parameters
    ----------
    W : sparse matrix
        Adjacency matrix.
    nodes : array-like, optional
        Node indices (if working on subgraph).

    Returns
    -------
    part0, part1 : lists
        Node indices in each partition.
    """
    if nodes is None:
        nodes = np.arange(W.shape[0])

    nodes_arr = np.asarray(nodes)
    n = len(nodes_arr)
    if n <= 1:
        return list(nodes_arr), []

    # Extract subgraph
    W_sub = W[nodes_arr, :][:, nodes_arr]

    # Build Laplacian for subgraph
    L_sub, _, _ = unweighted_laplacian(W_sub)

    # Handle disconnected components
    comps = connected_components(W_sub)

    part0_local = []
    part1_local = []

    for comp in comps:
        if len(comp) <= 1:
            # Single node goes to smaller partition
            if len(part0_local) <= len(part1_local):
                part0_local.extend(comp)
            else:
                part1_local.extend(comp)
            continue

        # Get subgraph for this component
        comp_arr = np.array(comp)
        L_comp = L_sub[comp_arr, :][:, comp_arr]

        # Compute Fiedler vector
        v = _fiedler_vector(L_comp, len(comp))

        # Split by median
        median_val = np.median(v)

        # Handle ties at median
        at_median = np.where(v == median_val)[0]
        if len(at_median) > 0:
            # Randomly perturb ties
            rng = np.random.default_rng(42)
            perturb = rng.standard_normal(len(at_median))
            for i, idx in enumerate(at_median):
                if perturb[i] > 0:
                    v[idx] = median_val + 1e-10
                else:
                    v[idx] = median_val - 1e-10

        # Partition
        for i, local_idx in enumerate(comp):
            if v[i] > median_val:
                part0_local.append(local_idx)
            else:
                part1_local.append(local_idx)

    # Map back to original node indices
    part0 = [nodes_arr[i] for i in part0_local]
    part1 = [nodes_arr[i] for i in part1_local]

    return part0, part1


def spectral_kway(A: Union[spmatrix, npt.ArrayLike], k: int) -> npt.NDArray[np.int32]:
    """
    Partition graph into k parts using recursive spectral bisection.

    Parameters
    ----------
    A : sparse matrix
        Input matrix (pattern defines graph).
    k : int
        Number of partitions (must be power of 2).

    Returns
    -------
    partition : ndarray
        Array of length n with partition labels (0 to k-1).
    """
    n = A.shape[0]

    if k < 1 or (k & (k - 1)) != 0:
        raise ValueError(f"k must be a power of 2, got {k}")

    if k == 1:
        return np.zeros(n, dtype=np.int32)

    # Get adjacency matrix
    _, W, _ = unweighted_laplacian(A)

    # Number of bisection levels
    num_levels = int(np.log2(k))

    # Initialize partition labels
    partition = np.zeros(n, dtype=np.int32)

    # Recursive bisection
    # Start with all nodes in partition 0
    current_partitions = {0: list(range(n))}

    for level in range(num_levels):
        new_partitions = {}
        for part_id, nodes in current_partitions.items():
            if len(nodes) <= 1:
                # Can't split further
                new_partitions[2 * part_id] = nodes
                new_partitions[2 * part_id + 1] = []
            else:
                nodes_arr = np.array(nodes)
                part0, part1 = _spectral_bisection(W, nodes_arr)
                new_partitions[2 * part_id] = part0
                new_partitions[2 * part_id + 1] = part1

        current_partitions = new_partitions

    # Assign final partition labels
    for part_id, nodes in current_partitions.items():
        for node in nodes:
            partition[node] = part_id

    # Renumber partitions to be contiguous 0 to k-1
    unique_parts = np.unique(partition)
    remap = {old: new for new, old in enumerate(unique_parts)}
    partition = np.array([remap[p] for p in partition], dtype=np.int32)

    return partition


def multilevel_partition(
    A: Union[spmatrix, npt.ArrayLike],
    nlev: int = 2,
    k: int = 4,
    minsep: int = 16,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], int]:
    """
    Create multilevel domain decomposition ordering.

    At each level:
    1. Partition the remaining nodes into k parts
    2. Identify interior nodes (no edges to other partitions)
    3. Identify separator nodes (have edges to other partitions)
    4. Interior nodes go to current level, separator forms next level

    Parameters
    ----------
    A : sparse matrix
        Input matrix.
    nlev : int, optional
        Target number of levels. Default is 2.
    k : int, optional
        Number of partitions per level (must be power of 2). Default is 4.
    minsep : int, optional
        Minimum separator size to continue partitioning. Default is 16.

    Returns
    -------
    p : ndarray
        Permutation array.
    lev_ptr : ndarray
        Level pointers. lev_ptr[i] is start of level i, lev_ptr[nlev] = n.
    nlev : int
        Actual number of levels (may be less than requested).
    """
    n = A.shape[0]

    if not issparse(A):
        A = csr_matrix(A)

    # Ensure k is at least 2 and power of 2
    if k < 2:
        k = 2
    minsep = max(minsep, k)

    # Build permutation by multilevel partitioning
    p = np.zeros(n, dtype=np.int64)
    ptr = 0
    remaining = list(range(n))
    lev_ptr = [0]
    current_level = 1

    while current_level < nlev and len(remaining) > minsep:
        remaining_arr = np.array(remaining)
        B = A[remaining_arr, :][:, remaining_arr]

        partition = spectral_kway(B, k)
        B_csr = B.tocsr()

        interior_local = []
        separator_local = []

        for local_idx in range(len(remaining)):
            my_part = partition[local_idx]

            row_start = B_csr.indptr[local_idx]
            row_end = B_csr.indptr[local_idx + 1]
            neighbors = B_csr.indices[row_start:row_end]

            is_separator = any(partition[nb] != my_part for nb in neighbors)

            if is_separator:
                separator_local.append(local_idx)
            else:
                interior_local.append(local_idx)

        # Map to global indices
        for local_idx in interior_local:
            p[ptr] = remaining_arr[local_idx]
            ptr += 1

        lev_ptr.append(ptr)

        # Update remaining to separator nodes
        remaining = [remaining_arr[i] for i in separator_local]
        current_level += 1

    # Add remaining nodes to final level
    for node in remaining:
        p[ptr] = node
        ptr += 1

    lev_ptr.append(n)

    return p, np.array(lev_ptr, dtype=np.int64), len(lev_ptr) - 1
