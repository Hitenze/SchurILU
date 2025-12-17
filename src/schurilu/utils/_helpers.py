"""
Utility functions for schurilu.
"""

from typing import Optional, Union, Any
import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse import csr_matrix, spmatrix, diags, eye, kron


def ensure_csr(A: Union[spmatrix, npt.ArrayLike]) -> csr_matrix:
    """
    Ensure the input matrix is in CSR format.

    Parameters
    ----------
    A : sparse matrix or array-like
        Input matrix.

    Returns
    -------
    A_csr : scipy.sparse.csr_matrix
        Matrix in CSR format.
    """
    if sparse.issparse(A):
        if not sparse.isspmatrix_csr(A):
            return A.tocsr()
        return A
    else:
        return sparse.csr_matrix(A)


def get_dtype(
    A: Union[spmatrix, npt.ArrayLike], dtype: Optional[npt.DTypeLike] = None
) -> np.dtype:
    """
    Determine the dtype to use for computation.

    Parameters
    ----------
    A : sparse matrix or array
        Input matrix.
    dtype : dtype, optional
        Explicit dtype. If None, uses A's dtype.

    Returns
    -------
    dtype : numpy dtype
        The dtype to use (always floating point).
    """
    if dtype is not None:
        return np.dtype(dtype)

    if sparse.issparse(A):
        input_dtype = A.dtype
    elif hasattr(A, "dtype"):
        input_dtype = A.dtype
    else:
        return np.dtype(np.float64)

    # Convert integer types to float64
    if np.issubdtype(input_dtype, np.integer):
        return np.dtype(np.float64)

    return np.dtype(input_dtype)


def is_complex_dtype(dtype: npt.DTypeLike) -> bool:
    """
    Check if a dtype is complex.

    Parameters
    ----------
    dtype : numpy dtype
        The dtype to check.

    Returns
    -------
    is_complex : bool
        True if dtype is complex.
    """
    return np.issubdtype(dtype, np.complexfloating)


def get_real_dtype(dtype: npt.DTypeLike) -> np.dtype:
    """
    Get the corresponding real dtype for a possibly complex dtype.

    Parameters
    ----------
    dtype : numpy dtype
        Input dtype (real or complex).

    Returns
    -------
    real_dtype : numpy dtype
        The corresponding real dtype.
    """
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        if dtype == np.complex64:
            return np.dtype(np.float32)
        else:
            return np.dtype(np.float64)
    return dtype


def get_complex_dtype(dtype: npt.DTypeLike) -> np.dtype:
    """
    Get the corresponding complex dtype for a real dtype.

    Parameters
    ----------
    dtype : numpy dtype
        Input dtype (real or complex).

    Returns
    -------
    complex_dtype : numpy dtype
        The corresponding complex dtype.
    """
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        return dtype
    if dtype == np.float32:
        return np.dtype(np.complex64)
    else:
        return np.dtype(np.complex128)


def fd3d(
    nx: int,
    ny: int = 1,
    nz: int = 1,
    shift: float = 0.0,
    dtype: npt.DTypeLike = np.float64,
) -> csr_matrix:
    """
    Construct a 3D finite difference Laplacian matrix.

    Creates the standard 7-point stencil discretization of the Laplacian
    operator on a regular nx x ny x nz grid with Dirichlet boundary conditions.

    Parameters
    ----------
    nx : int
        Number of grid points in x direction.
    ny : int, optional
        Number of grid points in y direction. Default is 1 (1D problem).
    nz : int, optional
        Number of grid points in z direction. Default is 1 (2D problem if ny > 1).
    shift : float, optional
        Diagonal shift. Subtracts shift * I from the matrix.
        Use to create indefinite problems (shift > 0 reduces diagonal dominance).
    dtype : numpy dtype, optional
        Data type of the matrix. Default is float64.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Sparse matrix of size (nx*ny*nz) x (nx*ny*nz).

    Examples
    --------
    >>> from schurilu.utils import fd3d
    >>> A = fd3d(10, 10, 10)  # 1000x1000 3D Laplacian
    >>> A = fd3d(10, 10, 10, shift=0.1)  # shifted (less diagonally dominant)
    >>> A = fd3d(10, 10, 1)  # 100x100 2D Laplacian
    """
    dtype = np.dtype(dtype)

    # 1D Laplacian
    def tridiag_1d(n: int) -> csr_matrix:
        return diags(
            [-np.ones(n - 1, dtype=dtype),
             2 * np.ones(n, dtype=dtype),
             -np.ones(n - 1, dtype=dtype)],
            [-1, 0, 1],
            format="csr",
        )

    tx = tridiag_1d(nx)
    Ix = eye(nx, format="csr", dtype=dtype)

    if ny == 1 and nz == 1:
        # 1D problem
        A = tx
    elif nz == 1:
        # 2D problem
        ty = tridiag_1d(ny)
        Iy = eye(ny, format="csr", dtype=dtype)
        A = kron(Iy, tx) + kron(ty, Ix)
    else:
        # 3D problem
        if ny == 1:
            raise ValueError("ny must be > 1 for 3D problems (nz > 1).")
        ty = tridiag_1d(ny)
        tz = tridiag_1d(nz)
        Iy = eye(ny, format="csr", dtype=dtype)
        Iz = eye(nz, format="csr", dtype=dtype)
        Ixy = eye(nx * ny, format="csr", dtype=dtype)
        A = kron(Iz, kron(Iy, tx)) + kron(Iz, kron(ty, Ix)) + kron(tz, Ixy)

    # Apply diagonal shift
    if shift != 0.0:
        n = nx * ny * nz
        A = A - shift * eye(n, format="csr", dtype=dtype)

    A = A.tocsr()
    A.eliminate_zeros()
    return A


def rlap2d(
    nx: int,
    ny: int,
    epsilon: float,
    theta: float,
    shift: float = 0.0,
    dtype: npt.DTypeLike = np.float64,
    D: float = 1.0,
) -> csr_matrix:
    """
    2D rotated anisotropic Laplacian (9-point) with Dirichlet BC.

    Parameters
    ----------
    nx, ny : int
        Number of grid points in x and y directions.
    epsilon : float
        Anisotropy ratio (>0).
    theta : float
        Rotation angle in radians.
    shift : float, optional
        Diagonal shift. Subtracts shift * I from the matrix.
    dtype : numpy dtype, optional
        Data type of the matrix. Default is float64.
    D : float, optional
        Diffusion scaling. Default is 1.0.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        (nx*ny) x (nx*ny) sparse matrix in unit scaling.
    """
    dtype = np.dtype(dtype)

    C = float(np.cos(theta))
    S = float(np.sin(theta))
    CC = C * C
    SS = S * S
    CS = C * S

    alpha = epsilon * CC + SS
    gamma = epsilon * SS + CC
    beta = (epsilon - 1.0) * CS

    c_n = -D * alpha
    c_s = -D * alpha
    c_w = -D * gamma
    c_e = -D * gamma
    c_ne = +0.5 * D * beta
    c_sw = +0.5 * D * beta
    c_nw = -0.5 * D * beta
    c_se = -0.5 * D * beta
    c_c = 2.0 * D * (alpha + gamma)

    n = nx * ny
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for j in range(ny):
        base = j * nx
        for i in range(nx):
            r = base + i
            rows.append(r); cols.append(r); data.append(c_c)

            if i > 0:
                rows.append(r); cols.append(r - 1); data.append(c_w)
            if i + 1 < nx:
                rows.append(r); cols.append(r + 1); data.append(c_e)
            if j > 0:
                rows.append(r); cols.append(r - nx); data.append(c_n)
            if j + 1 < ny:
                rows.append(r); cols.append(r + nx); data.append(c_s)

            if i + 1 < nx and j > 0:
                rows.append(r); cols.append(r - nx + 1); data.append(c_ne)
            if i > 0 and j > 0:
                rows.append(r); cols.append(r - nx - 1); data.append(c_nw)
            if i > 0 and j + 1 < ny:
                rows.append(r); cols.append(r + nx - 1); data.append(c_sw)
            if i + 1 < nx and j + 1 < ny:
                rows.append(r); cols.append(r + nx + 1); data.append(c_se)

    A = sparse.csr_matrix((np.asarray(data, dtype=dtype), (np.asarray(rows), np.asarray(cols))), shape=(n, n))

    if shift != 0.0:
        A = A - shift * eye(n, format="csr", dtype=dtype)

    A.eliminate_zeros()
    return A
