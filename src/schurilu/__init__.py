"""
SchurILU - Incomplete LU factorization with Schur complement support

A Python package for ILU preconditioners and Krylov solvers.
"""

from schurilu.preconditioners import ilu0, iluk, ilut, ILUResult, GeMSLR, arnoldi, fsai0, FSAI0Result
from schurilu.krylov import fgmres, fgmrez, pcg, planczos
from schurilu.reordering import (
    multilevel_partition,
    spectral_kway,
    unweighted_laplacian,
    connected_components,
)

__version__ = "0.1.1"

__all__ = [
    # Preconditioners
    "ilu0",
    "iluk",
    "ilut",
    "ILUResult",
    "fsai0",
    "FSAI0Result",
    # Krylov solvers
    "fgmres",
    "fgmrez",
    "pcg",
    "planczos",
    # GeMSLR preconditioner
    "GeMSLR",
    "arnoldi",
    # Reordering / partitioning
    "multilevel_partition",
    "spectral_kway",
    "unweighted_laplacian",
    "connected_components",
]
