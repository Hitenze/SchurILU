"""
ILU(0) factorization with Schur complement support.
"""

from typing import Optional, Union
import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse import csr_matrix, spmatrix

from schurilu.preconditioners._base import ILUResult
from schurilu.utils._helpers import ensure_csr, get_dtype


def ilu0(
    A: Union[spmatrix, npt.ArrayLike],
    nB: Optional[int] = None,
    modified: bool = False,
    dtype: Optional[npt.DTypeLike] = None,
) -> ILUResult:
    """
    Compute the ILU(0) factorization with optional Schur complement.

    ILU(0) computes an incomplete LU factorization using only the sparsity
    pattern of A (no fill-in allowed).

    For partial factorization (nB < n), the matrix is partitioned as:
        A = [B  F]
            [E  C]
    and only the B block is factored, with the Schur complement S returned.

    Parameters
    ----------
    A : sparse matrix
        Input matrix (will be converted to CSR format).
    nB : int, optional
        Size of the B block. Default is n (full factorization).
    modified : bool, optional
        If True, use modified ILU (dropped entries added to diagonal).
        Default is False.
    dtype : numpy dtype, optional
        Data type for computation. Default is A's dtype.

    Returns
    -------
    result : ILUResult
        Object containing L, D, U factors and optionally E, F, S.

    Notes
    -----
    The factorization is stored as:
    - L: Lower triangular without diagonal (unit diagonal implied)
    - D: Inverse of diagonal of U (1/d_ii)
    - U: Upper triangular without diagonal
    - E: E * U^{-1} block (when nB < n)
    - F: L^{-1} * F block (when nB < n)
    - S: Schur complement (when nB < n)

    Examples
    --------
    >>> from schurilu import ilu0
    >>> import scipy.sparse as sp
    >>> import numpy as np
    >>> A = sp.random(100, 100, density=0.1, format='csr')
    >>> A = A + 10 * sp.eye(100)  # Make diagonally dominant
    >>> result = ilu0(A)
    >>> b = np.random.randn(100)
    >>> x = result.solve(b)  # Apply preconditioner
    """
    # Convert to CSR and get dimensions
    A = ensure_csr(A)
    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square")

    if nB is None:
        nB = n

    if nB < 0 or nB > n:
        raise ValueError(f"nB must be between 0 and n={n}, got {nB}")

    # Determine dtype
    dtype = get_dtype(A, dtype)

    # Convert A to the desired dtype
    if A.dtype != dtype:
        A = A.astype(dtype)

    nS = n - nB
    zero_tol = (
        1e-14 if np.issubdtype(dtype, np.floating) and dtype == np.float64 else 1e-6
    )

    # Get CSR data (0-indexed)
    A_indptr = A.indptr
    A_indices = A.indices
    A_data = A.data

    # Estimate sizes for L, U, S
    nnz = A.nnz
    expand_fact = 1.25

    L_size = max(nB + nnz // 2, 1)
    U_size = L_size
    S_size = max(nS + nnz * nS // n, 1) if nS > 0 else 1

    # Allocate arrays for L
    L_indptr = np.zeros(n + 1, dtype=np.int64)
    L_indices = np.zeros(L_size, dtype=np.int64)
    L_data = np.zeros(L_size, dtype=dtype)

    # Allocate arrays for U
    U_indptr = np.zeros(n + 1, dtype=np.int64)
    U_indices = np.zeros(U_size, dtype=np.int64)
    U_data = np.zeros(U_size, dtype=dtype)

    # Allocate arrays for S
    S_indptr = np.zeros(nS + 1, dtype=np.int64)
    S_indices = np.zeros(S_size, dtype=np.int64)
    S_data = np.zeros(S_size, dtype=dtype)

    # Diagonal (stored as inverse)
    D = np.zeros(n, dtype=dtype)

    # Working arrays
    iw = -np.ones(n, dtype=np.int64)  # Index workspace
    iL = np.zeros(n, dtype=np.int64)  # Column indices for L
    wL = np.zeros(n, dtype=dtype)  # Values for L
    iU = np.zeros(n, dtype=np.int64)  # Column indices for U
    wU = np.zeros(n, dtype=dtype)  # Values for U

    L_ptr = 0
    U_ptr = 0
    S_ptr = 0

    # ===== B part factorization (rows 0 to nB-1) =====
    for i in range(nB):
        row_start = A_indptr[i]
        row_end = A_indptr[i + 1]

        drop = np.zeros(1, dtype=dtype)[0]  # For MILU
        dd = np.zeros(1, dtype=dtype)[0]  # Diagonal
        lenl = 0
        lenu = 0
        iw[i] = i

        # Copy row entries to working arrays
        for k in range(row_start, row_end):
            col = A_indices[k]
            val = A_data[k]

            if col < i:
                iw[col] = lenl
                iL[lenl] = col
                wL[lenl] = val
                lenl += 1
            elif col > i:
                iw[col] = lenu
                iU[lenu] = col
                wU[lenu] = val
                lenu += 1
            else:
                dd = val

        # Sort L entries by column index (for correct elimination order)
        if lenl > 0:
            order = np.argsort(iL[:lenl])
            iL[:lenl] = iL[:lenl][order]
            wL[:lenl] = wL[:lenl][order]
            for j in range(lenl):
                iw[iL[j]] = j

        # Elimination: for each L entry, update using U row
        for j in range(lenl):
            jpiv = iL[j]
            dpiv = wL[j] * D[jpiv]
            wL[j] = dpiv

            iw[jpiv] = -1

            # Loop through U[jpiv, :]
            U_row_start = U_indptr[jpiv]
            U_row_end = U_indptr[jpiv + 1]

            for k in range(U_row_start, U_row_end):
                col = U_indices[k]
                jpos = iw[col]

                if jpos < 0:
                    # Entry not in pattern - dropped
                    drop = drop - U_data[k] * dpiv
                    continue

                lxu = -U_data[k] * dpiv

                if col < i:
                    wL[jpos] = wL[jpos] + lxu
                elif col > i:
                    wU[jpos] = wU[jpos] + lxu
                else:
                    dd = dd + lxu

        # Reset iw
        iw[i] = -1
        for j in range(lenu):
            iw[iU[j]] = -1

        # Store L entries
        if lenl > 0:
            if L_ptr + lenl > L_size:
                # Expand L arrays
                while L_ptr + lenl > L_size:
                    L_size = int(L_size * expand_fact + 1)
                L_indices = np.resize(L_indices, L_size)
                L_data = np.resize(L_data, L_size)

            L_indices[L_ptr : L_ptr + lenl] = iL[:lenl]
            L_data[L_ptr : L_ptr + lenl] = wL[:lenl]
        L_ptr += lenl
        L_indptr[i + 1] = L_ptr

        # Apply MILU
        if modified:
            dd = dd + drop

        # Store diagonal (as inverse)
        if abs(dd) < zero_tol:
            # Preserve sign, default to positive
            if dd.real >= 0 or (dd.real == 0 and getattr(dd, "imag", 0) >= 0):
                dd = zero_tol
            else:
                dd = -zero_tol
        D[i] = 1.0 / dd

        # Store U entries
        if lenu > 0:
            if U_ptr + lenu > U_size:
                while U_ptr + lenu > U_size:
                    U_size = int(U_size * expand_fact + 1)
                U_indices = np.resize(U_indices, U_size)
                U_data = np.resize(U_data, U_size)

            U_indices[U_ptr : U_ptr + lenu] = iU[:lenu]
            U_data[U_ptr : U_ptr + lenu] = wU[:lenu]
        U_ptr += lenu
        U_indptr[i + 1] = U_ptr

    # ===== S part factorization (rows nB to n-1) =====
    for i in range(nB, n):
        row_start = A_indptr[i]
        row_end = A_indptr[i + 1]

        drop = np.zeros(1, dtype=dtype)[0]
        dd = np.zeros(1, dtype=dtype)[0]
        lenl = 0
        lenu = 0

        iw[i] = nB  # Marker

        # Copy row entries to working arrays
        for k in range(row_start, row_end):
            col = A_indices[k]
            val = A_data[k]

            if col < nB:
                # E part (goes into L)
                iw[col] = lenl
                iL[lenl] = col
                wL[lenl] = val
                lenl += 1
            elif col != i:
                # S part (off-diagonal)
                iw[col] = lenu
                iU[lenu] = col
                wU[lenu] = val
                lenu += 1
            else:
                # Diagonal of S
                dd = val

        # Sort L entries
        if lenl > 0:
            order = np.argsort(iL[:lenl])
            iL[:lenl] = iL[:lenl][order]
            wL[:lenl] = wL[:lenl][order]
            for j in range(lenl):
                iw[iL[j]] = j

        # Elimination
        for j in range(lenl):
            jpiv = iL[j]
            dpiv = wL[j] * D[jpiv]
            wL[j] = dpiv

            iw[jpiv] = -1

            U_row_start = U_indptr[jpiv]
            U_row_end = U_indptr[jpiv + 1]

            for k in range(U_row_start, U_row_end):
                col = U_indices[k]
                jpos = iw[col]

                if jpos < 0:
                    drop = drop - U_data[k] * dpiv
                    continue

                lxu = -U_data[k] * dpiv

                if col < nB:
                    wL[jpos] = wL[jpos] + lxu
                elif col != i:
                    wU[jpos] = wU[jpos] + lxu
                else:
                    dd = dd + lxu

        if modified:
            dd = dd + drop

        # Reset iw
        iw[i] = -1
        for k in range(lenu):
            iw[iU[k]] = -1

        # Store L entries (E * U^{-1} part)
        if lenl > 0:
            if L_ptr + lenl > L_size:
                while L_ptr + lenl > L_size:
                    L_size = int(L_size * expand_fact + 1)
                L_indices = np.resize(L_indices, L_size)
                L_data = np.resize(L_data, L_size)

            L_indices[L_ptr : L_ptr + lenl] = iL[:lenl]
            L_data[L_ptr : L_ptr + lenl] = wL[:lenl]
        L_ptr += lenl
        L_indptr[i + 1] = L_ptr

        # Store S entries (diagonal + off-diagonal)
        si = i - nB  # S index

        # Need space for diagonal + off-diagonal
        if S_ptr + lenu + 1 > S_size:
            while S_ptr + lenu + 1 > S_size:
                S_size = int(S_size * expand_fact + 1)
            S_indices = np.resize(S_indices, S_size)
            S_data = np.resize(S_data, S_size)

        # Store diagonal of S
        S_indices[S_ptr] = si
        S_data[S_ptr] = dd
        S_ptr += 1

        # Store off-diagonal of S
        if lenu > 0:
            S_indices[S_ptr : S_ptr + lenu] = iU[:lenu] - nB
            S_data[S_ptr : S_ptr + lenu] = wU[:lenu]
        S_ptr += lenu

        S_indptr[si + 1] = S_ptr

    # Set U_indptr for S part rows (no U entries for rows >= nB)
    if nB < n:
        U_indptr[nB + 1 :] = U_ptr

    # Trim arrays
    L_indices = L_indices[:L_ptr]
    L_data = L_data[:L_ptr]
    U_indices = U_indices[:U_ptr]
    U_data = U_data[:U_ptr]
    S_indices = S_indices[:S_ptr]
    S_data = S_data[:S_ptr]

    # Create sparse matrices
    L = csr_matrix((L_data, L_indices, L_indptr), shape=(n, n), dtype=dtype)
    U = csr_matrix((U_data, U_indices, U_indptr), shape=(n, n), dtype=dtype)

    # Create E, F, S if partial factorization
    if nB < n:
        S = csr_matrix((S_data, S_indices, S_indptr), shape=(nS, nS), dtype=dtype)

        # E is the lower-left part of L (rows nB:n, cols 0:nB)
        E = L[nB:, :nB].tocsr()

        # F is the upper-right part of U (rows 0:nB, cols nB:n)
        F = U[:nB, nB:].tocsr()
    else:
        E = None
        F = None
        S = None

    return ILUResult(L, D, U, E=E, F=F, S=S, n=n, nB=nB)
