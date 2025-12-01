"""
ILUT factorization with threshold-based dropping and Schur complement support.
"""

from typing import Optional, Union, List, Tuple
import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse import csr_matrix, spmatrix
import heapq

from schurilu.preconditioners._base import ILUResult
from schurilu.utils._helpers import ensure_csr, get_dtype


def ilut(
    A: Union[spmatrix, npt.ArrayLike],
    droptol: Union[float, List[float], Tuple[float, ...]] = 1e-2,
    lfil: Union[int, List[int], Tuple[int, ...]] = 100,
    nB: Optional[int] = None,
    modified: bool = False,
    dtype: Optional[npt.DTypeLike] = None,
) -> ILUResult:
    """
    Compute the ILUT factorization with optional Schur complement.

    ILUT computes an incomplete LU factorization using threshold-based dropping.
    Entries with |value| < droptol * ||row|| are dropped, and the number of
    entries per row is limited by lfil.

    Parameters
    ----------
    A : sparse matrix
        Input matrix (will be converted to CSR format).
    droptol : float or list, optional
        Drop tolerance. Entries with |value| < droptol * ||row|| are dropped.
        Can be:
        - float: Same tolerance for all parts (default 1e-2)
        - [drolb, drolef, drols]: Different tolerances for B, E/F, and S parts
    lfil : int or list, optional
        Maximum fill per row. Can be:
        - int: Same limit for all parts (default 100)
        - [lfilb, lfilef, lfils]: Different limits for B, E/F, and S parts
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

    # Parse droptol
    if isinstance(droptol, (list, tuple)):
        if len(droptol) >= 3:
            drolb, drolef, drols = droptol[0], droptol[1], droptol[2]
        elif len(droptol) == 2:
            drolb, drolef, drols = droptol[0], droptol[1], droptol[0]
        else:
            drolb = drolef = drols = droptol[0]
    else:
        drolb = drolef = drols = droptol

    # Parse lfil
    if isinstance(lfil, (list, tuple)):
        if len(lfil) >= 3:
            lfilb, lfilef, lfils = lfil[0], lfil[1], lfil[2]
        elif len(lfil) == 2:
            lfilb, lfilef, lfils = lfil[0], lfil[1], lfil[0]
        else:
            lfilb = lfilef = lfils = lfil[0]
    else:
        lfilb = lfilef = lfils = lfil

    # Determine dtype
    dtype = get_dtype(A, dtype)

    # Convert A to the desired dtype
    if A.dtype != dtype:
        A = A.astype(dtype)

    nS = n - nB
    zero_tol = (
        1e-14 if np.issubdtype(dtype, np.floating) and dtype == np.float64 else 1e-6
    )

    # Get CSR data
    A_indptr = A.indptr
    A_indices = A.indices
    A_data = A.data

    # Estimate sizes
    nnz = A.nnz
    expand_fact = 1.25

    L_size = max(nB * lfilb + nS * lfilef, nnz, 1)
    U_size = L_size
    S_size = max(nS * (lfils + 1), 1) if nS > 0 else 1

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
    iw = -np.ones(n, dtype=np.int64)
    w = np.zeros(n, dtype=dtype)

    # Store U data for elimination
    U_values = [{} for _ in range(n)]

    L_ptr = 0
    U_ptr = 0
    S_ptr = 0

    # ===== B part factorization (rows 0 to nB-1) =====
    for i in range(nB):
        row_start = A_indptr[i]
        row_end = A_indptr[i + 1]

        drop = np.zeros(1, dtype=dtype)[0]
        dd = np.zeros(1, dtype=dtype)[0]
        lenl = 0
        lenu = 0

        # Compute row norm for dropping
        row_norm = np.zeros(1, dtype=np.float64)[0]
        for k in range(row_start, row_end):
            row_norm += np.abs(A_data[k]) ** 2
        row_norm = np.sqrt(row_norm)
        if row_norm == 0:
            row_norm = 1.0

        # Copy row entries to working arrays
        iL = np.zeros(n, dtype=np.int64)
        iU = np.zeros(n, dtype=np.int64)

        for k in range(row_start, row_end):
            col = A_indices[k]
            val = A_data[k]

            if col < i:
                iw[col] = lenl
                iL[lenl] = col
                w[col] = val
                lenl += 1
            elif col > i:
                iw[col] = lenu
                iU[lenu] = col
                w[col] = val
                lenu += 1
            else:
                dd = val

        # Use heap to process L entries in column order
        l_heap = list(iL[:lenl])
        heapq.heapify(l_heap)
        processed = set()

        # Elimination: process L entries in column order using heap
        while l_heap:
            jpiv = heapq.heappop(l_heap)
            if jpiv in processed:
                continue
            processed.add(jpiv)

            dpiv = w[jpiv] * D[jpiv]
            w[jpiv] = dpiv

            # Loop through U[jpiv, :]
            for ucol, uval in U_values[jpiv].items():
                if iw[ucol] >= 0:
                    # Already in pattern
                    w[ucol] -= uval * dpiv
                elif ucol == i:
                    dd -= uval * dpiv
                elif ucol < i:
                    # Fill-in in L
                    iw[ucol] = lenl
                    iL[lenl] = ucol
                    w[ucol] = -uval * dpiv
                    lenl += 1
                    heapq.heappush(l_heap, ucol)
                else:
                    # Fill-in in U (ucol > i)
                    iw[ucol] = lenu
                    iU[lenu] = ucol
                    w[ucol] = -uval * dpiv
                    lenu += 1

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

        # Apply dropping and lfil limit for L
        drol_l = drolb
        lfil_l = lfilb
        tol_l = drol_l * row_norm

        # Collect L entries with values
        l_entries = []
        for j in range(lenl):
            col = iL[j]
            val = w[col]
            if np.abs(val) >= tol_l:
                l_entries.append((col, val, np.abs(val)))

        # Sort by magnitude (descending) and keep top lfil
        l_entries.sort(key=lambda x: -x[2])
        l_entries = l_entries[:lfil_l]

        # Sort by column for storage
        l_entries.sort(key=lambda x: x[0])

        # Store L entries
        final_lenl = len(l_entries)
        if final_lenl > 0:
            if L_ptr + final_lenl > L_size:
                while L_ptr + final_lenl > L_size:
                    L_size = int(L_size * expand_fact + 1)
                L_indices = np.resize(L_indices, L_size)
                L_data = np.resize(L_data, L_size)

            for idx, (col, val, _) in enumerate(l_entries):
                L_indices[L_ptr + idx] = col
                L_data[L_ptr + idx] = val
        L_ptr += final_lenl
        L_indptr[i + 1] = L_ptr

        # Apply dropping and lfil limit for U
        drol_u = drolb if i < nB else drolef
        lfil_u = lfilb
        tol_u = drol_u * row_norm

        # Collect U entries with values
        u_entries = []
        for j in range(lenu):
            col = iU[j]
            val = w[col]
            # Use different limits for F part (cols >= nB)
            if col >= nB:
                tol_check = drolef * row_norm
                lfil_check = lfilef
            else:
                tol_check = drolb * row_norm
                lfil_check = lfilb
            if np.abs(val) >= tol_check:
                u_entries.append((col, val, np.abs(val)))

        # Sort by magnitude (descending) and keep top lfil_u
        u_entries.sort(key=lambda x: -x[2])
        u_entries = u_entries[:lfil_u]

        # Sort by column for storage
        u_entries.sort(key=lambda x: x[0])

        # Store U entries
        final_lenu = len(u_entries)
        if final_lenu > 0:
            if U_ptr + final_lenu > U_size:
                while U_ptr + final_lenu > U_size:
                    U_size = int(U_size * expand_fact + 1)
                U_indices = np.resize(U_indices, U_size)
                U_data = np.resize(U_data, U_size)

            for idx, (col, val, _) in enumerate(u_entries):
                U_indices[U_ptr + idx] = col
                U_data[U_ptr + idx] = val
                U_values[i][col] = val
        U_ptr += final_lenu
        U_indptr[i + 1] = U_ptr

        # Reset working arrays
        for j in range(lenl):
            col = iL[j]
            iw[col] = -1
            w[col] = np.zeros(1, dtype=dtype)[0]
        for j in range(lenu):
            col = iU[j]
            iw[col] = -1
            w[col] = np.zeros(1, dtype=dtype)[0]

    # ===== S part factorization (rows nB to n-1) =====
    for i in range(nB, n):
        row_start = A_indptr[i]
        row_end = A_indptr[i + 1]

        drop = np.zeros(1, dtype=dtype)[0]
        dd = np.zeros(1, dtype=dtype)[0]
        lenl = 0  # E part (cols < nB)
        lenu = 0  # S part off-diagonals

        # Compute row norm for dropping
        row_norm = np.zeros(1, dtype=np.float64)[0]
        for k in range(row_start, row_end):
            row_norm += np.abs(A_data[k]) ** 2
        row_norm = np.sqrt(row_norm)
        if row_norm == 0:
            row_norm = 1.0

        iL = np.zeros(n, dtype=np.int64)
        iU = np.zeros(n, dtype=np.int64)

        # Copy row entries to working arrays
        for k in range(row_start, row_end):
            col = A_indices[k]
            val = A_data[k]

            if col < nB:
                # E part (goes into L)
                iw[col] = lenl
                iL[lenl] = col
                w[col] = val
                lenl += 1
            elif col != i:
                # S part (off-diagonal)
                iw[col] = lenu
                iU[lenu] = col
                w[col] = val
                lenu += 1
            else:
                # Diagonal of S
                dd = val

        # Use heap to process L entries in column order
        l_heap = list(iL[:lenl])
        heapq.heapify(l_heap)
        processed = set()

        # Elimination
        while l_heap:
            jpiv = heapq.heappop(l_heap)
            if jpiv in processed:
                continue
            processed.add(jpiv)

            dpiv = w[jpiv] * D[jpiv]
            w[jpiv] = dpiv

            for ucol, uval in U_values[jpiv].items():
                if iw[ucol] >= 0:
                    w[ucol] -= uval * dpiv
                elif ucol == i:
                    dd -= uval * dpiv
                elif ucol < nB:
                    # Fill-in in E part
                    iw[ucol] = lenl
                    iL[lenl] = ucol
                    w[ucol] = -uval * dpiv
                    lenl += 1
                    heapq.heappush(l_heap, ucol)
                elif ucol != i:
                    # Fill-in in S part
                    iw[ucol] = lenu
                    iU[lenu] = ucol
                    w[ucol] = -uval * dpiv
                    lenu += 1

        if modified:
            dd = dd + drop

        # Apply dropping and lfil limit for L (E part)
        tol_l = drolef * row_norm
        lfil_l = lfilef

        l_entries = []
        for j in range(lenl):
            col = iL[j]
            val = w[col]
            if np.abs(val) >= tol_l:
                l_entries.append((col, val, np.abs(val)))

        l_entries.sort(key=lambda x: -x[2])
        l_entries = l_entries[:lfil_l]
        l_entries.sort(key=lambda x: x[0])

        final_lenl = len(l_entries)
        if final_lenl > 0:
            if L_ptr + final_lenl > L_size:
                while L_ptr + final_lenl > L_size:
                    L_size = int(L_size * expand_fact + 1)
                L_indices = np.resize(L_indices, L_size)
                L_data = np.resize(L_data, L_size)

            for idx, (col, val, _) in enumerate(l_entries):
                L_indices[L_ptr + idx] = col
                L_data[L_ptr + idx] = val
        L_ptr += final_lenl
        L_indptr[i + 1] = L_ptr

        # Apply dropping and lfil limit for S off-diagonals
        tol_s = drols * row_norm
        lfil_s = lfils

        s_entries = []
        for j in range(lenu):
            col = iU[j]
            val = w[col]
            if np.abs(val) >= tol_s:
                s_entries.append((col, val, np.abs(val)))

        s_entries.sort(key=lambda x: -x[2])
        s_entries = s_entries[:lfil_s]
        s_entries.sort(key=lambda x: x[0])

        # Store S entries (diagonal + off-diagonal)
        si = i - nB

        final_lenu = len(s_entries)
        if S_ptr + final_lenu + 1 > S_size:
            while S_ptr + final_lenu + 1 > S_size:
                S_size = int(S_size * expand_fact + 1)
            S_indices = np.resize(S_indices, S_size)
            S_data = np.resize(S_data, S_size)

        # Store diagonal of S
        S_indices[S_ptr] = si
        S_data[S_ptr] = dd
        S_ptr += 1

        # Store off-diagonal of S
        for idx, (col, val, _) in enumerate(s_entries):
            S_indices[S_ptr + idx] = col - nB
            S_data[S_ptr + idx] = val
        S_ptr += final_lenu

        S_indptr[si + 1] = S_ptr

        # Reset working arrays
        for j in range(lenl):
            col = iL[j]
            iw[col] = -1
            w[col] = np.zeros(1, dtype=dtype)[0]
        for j in range(lenu):
            col = iU[j]
            iw[col] = -1
            w[col] = np.zeros(1, dtype=dtype)[0]

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
