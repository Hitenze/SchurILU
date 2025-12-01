"""
Reordering module for matrix permutations and graph partitioning.

This module provides spectral graph partitioning methods for creating
multilevel domain decomposition orderings.
"""

from schurilu.reordering.spectral import (
    unweighted_laplacian,
    connected_components,
    spectral_kway,
    multilevel_partition,
)

__all__ = [
    "unweighted_laplacian",
    "connected_components",
    "spectral_kway",
    "multilevel_partition",
]
