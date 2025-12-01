"""
Preconditioned Lanczos for eigenvalue estimation.
"""

from typing import Optional, Union, Callable, Any
import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import spmatrix
from scipy.sparse.linalg import LinearOperator

# Type alias for preconditioner
Preconditioner = Union[
    LinearOperator, Callable[[npt.NDArray[Any]], npt.NDArray[Any]], Any
]


def planczos(
    A: Union[spmatrix, LinearOperator, Callable[[npt.NDArray[Any]], npt.NDArray[Any]]],
    v0: npt.NDArray[Any],
    M: Optional[Preconditioner] = None,
    m: int = 50,
) -> npt.NDArray[Any]:
    """
    Preconditioned Lanczos for estimating eigenvalues of M^{-1} * A.

    This is useful for estimating the condition number of the preconditioned
    system and for choosing optimal Krylov method parameters.

    Parameters
    ----------
    A : sparse matrix or LinearOperator
        The system matrix.
    v0 : ndarray
        Starting vector (will be normalized).
    M : LinearOperator or callable, optional
        Preconditioner. Can be:
        - LinearOperator with matvec method
        - Callable that takes a vector and returns M^{-1} * v
        - Object with solve() method (like ILUResult)
        Default is identity (eigenvalues of A).
    m : int, optional
        Number of Lanczos iterations. Default is 50.

    Returns
    -------
    eigs : ndarray
        Estimated eigenvalues of M^{-1} * A (or A if M is None).

    Notes
    -----
    The Lanczos algorithm builds a tridiagonal matrix T whose eigenvalues
    approximate the eigenvalues of M^{-1} * A. The approximation improves
    with more iterations.

    For SPD matrices, the eigenvalues are real and positive. The extreme
    eigenvalues (largest and smallest) converge fastest.
    """
    n = len(v0)
    dtype = v0.dtype

    # Get matrix-vector product
    if hasattr(A, 'matvec'):
        matvec = A.matvec
    elif hasattr(A, '__matmul__'):
        matvec = lambda v: A @ v
    else:
        matvec = A

    # Get preconditioner application
    if M is None:
        psolve = lambda v: v.copy()
    elif hasattr(M, 'solve'):
        psolve = M.solve
    elif hasattr(M, 'matvec'):
        psolve = M.matvec
    elif callable(M):
        psolve = M
    else:
        psolve = lambda v: v.copy()

    # Normalize starting vector
    v = v0.copy()
    v = v / np.linalg.norm(v)

    # Storage for Lanczos vectors
    V = np.zeros((n, m + 1), dtype=dtype)
    V[:, 0] = v

    # Tridiagonal matrix elements
    alpha = np.zeros(m, dtype=dtype)
    beta = np.zeros(m, dtype=dtype)

    v_prev = np.zeros(n, dtype=dtype)
    beta_prev = 0.0

    for j in range(m):
        # Matrix-vector product: w = A * v
        w = matvec(V[:, j])

        # Apply preconditioner: z = M^{-1} * w
        z = psolve(w)

        # Lanczos recurrence
        alpha[j] = np.dot(V[:, j], z)

        # Orthogonalize
        w = z - alpha[j] * V[:, j]
        if j > 0:
            w = w - beta[j - 1] * V[:, j - 1]

        # Reorthogonalize (full reorthogonalization for stability)
        for i in range(j + 1):
            h = np.dot(w, V[:, i])
            w = w - h * V[:, i]

        beta[j] = np.linalg.norm(w)

        # Check for breakdown
        if beta[j] < 1e-14:
            # Early termination - invariant subspace found
            m = j + 1
            break

        V[:, j + 1] = w / beta[j]

    # Compute eigenvalues of tridiagonal matrix
    if m == 1:
        return np.array([alpha[0]], dtype=dtype)

    # Use scipy's specialized tridiagonal eigenvalue solver
    d = alpha[:m].real.astype(np.float64)
    e = beta[:m - 1].real.astype(np.float64)

    try:
        eigs = eigh_tridiagonal(d, e, eigvals_only=True)
    except Exception:
        # Fallback to general eigenvalue solver
        T = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        eigs = np.linalg.eigvalsh(T)

    return eigs.astype(dtype)
