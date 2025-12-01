"""
Preconditioned Conjugate Gradient solver.
"""

from typing import Optional, Union, Callable, Tuple, Any
import numpy as np
import numpy.typing as npt
from scipy.sparse import spmatrix
from scipy.sparse.linalg import LinearOperator

# Type alias for preconditioner
Preconditioner = Union[
    LinearOperator, Callable[[npt.NDArray[Any]], npt.NDArray[Any]], Any
]


def pcg(
    A: Union[spmatrix, LinearOperator, Callable[[npt.NDArray[Any]], npt.NDArray[Any]]],
    b: npt.NDArray[Any],
    x0: Optional[npt.NDArray[Any]] = None,
    M: Optional[Preconditioner] = None,
    tol: float = 1e-6,
    maxiter: int = 1000,
    atol: Optional[float] = None,
    callback: Optional[Callable[[float], None]] = None,
) -> Tuple[npt.NDArray[Any], int]:
    """
    Preconditioned Conjugate Gradient for solving Ax = b.

    The matrix A must be symmetric positive definite (SPD).
    The preconditioner M should approximate A^{-1}.

    Parameters
    ----------
    A : sparse matrix or LinearOperator
        The system matrix (must be SPD).
    b : ndarray
        Right-hand side vector.
    x0 : ndarray, optional
        Initial guess. Default is zero vector.
    M : LinearOperator or callable, optional
        Preconditioner. Can be:
        - LinearOperator with matvec method
        - Callable that takes a vector and returns M^{-1} * v
        - Object with solve() method (like ILUResult)
        Default is identity (no preconditioning).
    tol : float, optional
        Relative tolerance for convergence. Default is 1e-6.
    maxiter : int, optional
        Maximum number of iterations. Default is 1000.
    atol : float, optional
        Absolute tolerance. If None, uses tol * ||b||.
    callback : callable, optional
        Called after each iteration with the current residual norm.

    Returns
    -------
    x : ndarray
        The solution vector.
    info : int
        0: successful convergence
        >0: did not converge, info = number of iterations
    """
    n = len(b)

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

    # Initial guess
    if x0 is None:
        x = np.zeros(n, dtype=b.dtype)
    else:
        x = x0.copy()

    # Compute initial residual
    r = b - matvec(x)
    bnorm = np.linalg.norm(b)

    if bnorm == 0:
        bnorm = 1.0

    # Determine tolerance
    if atol is None:
        atol = tol * bnorm

    # Check initial convergence
    rnorm = np.linalg.norm(r)
    if rnorm <= atol or rnorm <= tol * bnorm:
        return x, 0

    # Apply preconditioner
    z = psolve(r)
    p = z.copy()
    rho = np.dot(r, z)

    for k in range(maxiter):
        # Matrix-vector product
        Ap = matvec(p)

        # Step length
        alpha = rho / np.dot(p, Ap)

        # Update solution and residual
        x = x + alpha * p
        r = r - alpha * Ap

        # Compute residual norm
        rnorm = np.linalg.norm(r)

        if callback is not None:
            callback(rnorm)

        # Check convergence
        if rnorm <= atol or rnorm <= tol * bnorm:
            return x, 0

        # Apply preconditioner
        z = psolve(r)

        # New rho
        rho_new = np.dot(r, z)

        # Direction update
        beta = rho_new / rho
        p = z + beta * p
        rho = rho_new

    # Did not converge
    return x, k + 1
