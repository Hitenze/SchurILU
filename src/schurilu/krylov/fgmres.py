"""
Flexible GMRES (real) solver.
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


def fgmres(
    A: Union[spmatrix, LinearOperator, Callable[[npt.NDArray[Any]], npt.NDArray[Any]]],
    b: npt.NDArray[Any],
    x0: Optional[npt.NDArray[Any]] = None,
    M: Optional[Preconditioner] = None,
    tol: float = 1e-6,
    maxiter: int = 1000,
    restart: int = 20,
    atol: Optional[float] = None,
    callback: Optional[Callable[[float], None]] = None,
) -> Tuple[npt.NDArray[Any], int]:
    """
    Flexible GMRES for solving Ax = b (real arithmetic).

    FGMRES allows the preconditioner M to vary at each iteration,
    which is useful for nonlinear preconditioners.

    Parameters
    ----------
    A : sparse matrix or LinearOperator
        The system matrix.
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
    restart : int, optional
        Number of iterations between restarts. Default is 20.
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

    # Total iterations counter
    total_iter = 0

    # Outer loop (restarts)
    while total_iter < maxiter:
        # Compute residual norm
        rnorm = np.linalg.norm(r)

        # Check convergence (before any matvec in this restart cycle)
        if rnorm <= atol or rnorm <= tol * bnorm:
            return x, 0

        # Normalize residual
        beta = rnorm

        # Allocate Arnoldi vectors and preconditioned vectors
        m = min(restart, maxiter - total_iter)
        V = np.zeros((n, m + 1), dtype=b.dtype)
        Z = np.zeros((n, m), dtype=b.dtype)
        H = np.zeros((m + 1, m), dtype=b.dtype)

        V[:, 0] = r / beta

        # Givens rotation parameters
        cs = np.zeros(m, dtype=b.dtype)
        sn = np.zeros(m, dtype=b.dtype)
        g = np.zeros(m + 1, dtype=b.dtype)
        g[0] = beta

        # Inner loop (Arnoldi)
        j = 0
        for j in range(m):
            # Apply preconditioner
            Z[:, j] = psolve(V[:, j])

            # Matrix-vector product
            w = matvec(Z[:, j])

            # Modified Gram-Schmidt orthogonalization
            for i in range(j + 1):
                H[i, j] = np.dot(w, V[:, i])
                w = w - H[i, j] * V[:, i]

            H[j + 1, j] = np.linalg.norm(w)

            # Check for breakdown
            if H[j + 1, j] < 1e-14:
                m = j + 1
                break

            V[:, j + 1] = w / H[j + 1, j]

            # Apply previous Givens rotations to the new column
            for i in range(j):
                temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = temp

            # Compute new Givens rotation
            denom = np.sqrt(H[j, j]**2 + H[j + 1, j]**2)
            cs[j] = H[j, j] / denom
            sn[j] = H[j + 1, j] / denom

            # Apply Givens rotation to H and g
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0

            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]

            total_iter += 1

            # Callback with current residual estimate
            if callback is not None:
                callback(abs(g[j + 1]))

            # Check convergence
            if abs(g[j + 1]) <= atol or abs(g[j + 1]) <= tol * bnorm:
                # Solve the upper triangular system
                y = np.zeros(j + 1, dtype=b.dtype)
                for i in range(j, -1, -1):
                    y[i] = g[i]
                    for k in range(i + 1, j + 1):
                        y[i] -= H[i, k] * y[k]
                    y[i] /= H[i, i]

                # Update solution
                x = x + Z[:, :j + 1] @ y
                return x, 0

        # Solve the upper triangular system
        j_end = j + 1
        y = np.zeros(j_end, dtype=b.dtype)
        for i in range(j_end - 1, -1, -1):
            y[i] = g[i]
            for k in range(i + 1, j_end):
                y[i] -= H[i, k] * y[k]
            y[i] /= H[i, i]

        # Update solution
        x = x + Z[:, :j_end] @ y

        # Compute new residual
        r = b - matvec(x)

    # Did not converge
    return x, total_iter
