"""
ILU(k) factorization with level-based fill-in and Schur complement support.
"""

from typing import Optional, Union, List, Tuple
import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse import csr_matrix, spmatrix
import heapq

from schurilu.preconditioners._base import ILUResult
from schurilu.utils._helpers import ensure_csr, get_dtype


def iluk(
    A: Union[spmatrix, npt.ArrayLike],
    lfil: Union[int, List[int], Tuple[int, ...]] = 1,
    nB: Optional[int] = None,
    modified: bool = False,
    dtype: Optional[npt.DTypeLike] = None,
) -> ILUResult:
    """
    Compute the ILU(k) factorization with optional Schur complement.

    ILU(k) computes an incomplete LU factorization using level-based fill-in.
    Level 0 entries are those in the original pattern of A. Level k entries
    are created when multiplying level i and level j entries where i+j+1 = k.

    Parameters
    ----------
    A : sparse matrix
        Input matrix (will be converted to CSR format).
    lfil : int or list, optional
        Fill level. Can be:
        - int: Same level for all parts (default 1)
        - [lfilb, lfilef, lfils]: Different levels for B, E/F, and S parts
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

    # Level arrays
    levl = np.full(n, -1, dtype=np.int32)
    levu = np.full(n, -1, dtype=np.int32)

    # Working arrays
    iw = -np.ones(n, dtype=np.int64)
    iL = np.zeros(n, dtype=np.int64)
    wL = np.zeros(n, dtype=dtype)
    iU = np.zeros(n, dtype=np.int64)
    wU = np.zeros(n, dtype=dtype)

    # Store U levels and data for symbolic access
    U_levels = [[] for _ in range(n)]  # U_levels[i] = [(col, level), ...]
    U_values = [{} for _ in range(n)]  # U_values[i] = {col: value, ...}

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
        iw[i] = i

        # Copy row entries to working arrays
        for k in range(row_start, row_end):
            col = A_indices[k]
            val = A_data[k]

            if col < i:
                iw[col] = lenl
                levl[col] = 0
                iL[lenl] = col
                wL[lenl] = val
                lenl += 1
            elif col > i:
                iw[col] = lenu
                levu[col] = 0
                iU[lenu] = col
                wU[lenu] = val
                lenu += 1
            else:
                dd = val

        # Process L entries with heap for level-based fill
        # Build heap of (col, level) pairs
        l_heap = [(iL[j], 0) for j in range(lenl)]
        heapq.heapify(l_heap)
        processed = set()

        while l_heap:
            jpiv, lev_jpiv = heapq.heappop(l_heap)
            if jpiv in processed:
                continue
            if lev_jpiv > lfilb:
                levl[jpiv] = -1
                continue
            processed.add(jpiv)

            # Look at U[jpiv, :] to create fill-in
            for ucol, ulev in U_levels[jpiv]:
                new_lev = lev_jpiv + ulev + 1

                if ucol < i:
                    # Fill in L
                    if new_lev <= lfilb:
                        if iw[ucol] < 0:
                            # New fill-in
                            iw[ucol] = lenl
                            levl[ucol] = new_lev
                            iL[lenl] = ucol
                            wL[lenl] = np.zeros(1, dtype=dtype)[0]
                            lenl += 1
                            heapq.heappush(l_heap, (ucol, new_lev))
                        elif levl[ucol] > new_lev:
                            levl[ucol] = new_lev
                            heapq.heappush(l_heap, (ucol, new_lev))
                elif ucol > i:
                    # Fill in U
                    lfil_check = lfilb if ucol < nB else lfilef
                    if new_lev <= lfil_check:
                        if iw[ucol] < 0:
                            iw[ucol] = lenu
                            levu[ucol] = new_lev
                            iU[lenu] = ucol
                            wU[lenu] = np.zeros(1, dtype=dtype)[0]
                            lenu += 1
                        elif levu[ucol] > new_lev:
                            levu[ucol] = new_lev

        # Sort L entries by column index
        if lenl > 0:
            order = np.argsort(iL[:lenl])
            iL[:lenl] = iL[:lenl][order]
            wL[:lenl] = wL[:lenl][order]
            levl_temp = np.array([levl[iL[j]] for j in range(lenl)])
            for j in range(lenl):
                iw[iL[j]] = j
                levl[iL[j]] = levl_temp[j]

        # Elimination: for each L entry, update using U row
        for j in range(lenl):
            jpiv = iL[j]
            if levl[jpiv] < 0 or levl[jpiv] > lfilb:
                continue
            dpiv = wL[j] * D[jpiv]
            wL[j] = dpiv

            # Loop through U[jpiv, :]
            for ucol, uval in U_values[jpiv].items():
                jpos = iw[ucol]

                if jpos < 0:
                    # Entry not in pattern - dropped
                    drop = drop - uval * dpiv
                    continue

                lxu = -uval * dpiv

                if ucol < i:
                    wL[jpos] = wL[jpos] + lxu
                elif ucol > i:
                    wU[jpos] = wU[jpos] + lxu
                else:
                    dd = dd + lxu

        # Reset iw for cols we won't keep
        iw[i] = -1

        # Filter and store L entries
        final_lenl = 0
        for j in range(lenl):
            col = iL[j]
            if levl[col] >= 0 and levl[col] <= lfilb:
                iL[final_lenl] = col
                wL[final_lenl] = wL[j]
                final_lenl += 1
            iw[col] = -1
            levl[col] = -1
        lenl = final_lenl

        # Filter and store U entries
        final_lenu = 0
        for j in range(lenu):
            col = iU[j]
            lfil_check = lfilb if col < nB else lfilef
            if levu[col] >= 0 and levu[col] <= lfil_check:
                iU[final_lenu] = col
                wU[final_lenu] = wU[j]
                final_lenu += 1
            iw[col] = -1
            levu[col] = -1
        lenu = final_lenu

        # Store L entries
        if lenl > 0:
            if L_ptr + lenl > L_size:
                while L_ptr + lenl > L_size:
                    L_size = int(L_size * expand_fact + 1)
                L_indices = np.resize(L_indices, L_size)
                L_data = np.resize(L_data, L_size)

            # Sort by column before storing
            order = np.argsort(iL[:lenl])
            L_indices[L_ptr : L_ptr + lenl] = iL[:lenl][order]
            L_data[L_ptr : L_ptr + lenl] = wL[:lenl][order]
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

            # Sort by column before storing
            order = np.argsort(iU[:lenu])
            sorted_cols = iU[:lenu][order]
            sorted_vals = wU[:lenu][order]
            U_indices[U_ptr : U_ptr + lenu] = sorted_cols
            U_data[U_ptr : U_ptr + lenu] = sorted_vals

            # Store levels and values for symbolic access
            for j in range(lenu):
                col = sorted_cols[j]
                U_levels[i].append((col, 0))  # Original entries have level 0
                U_values[i][col] = sorted_vals[j]

        U_ptr += lenu
        U_indptr[i + 1] = U_ptr

    # ===== S part factorization (rows nB to n-1) =====
    for i in range(nB, n):
        row_start = A_indptr[i]
        row_end = A_indptr[i + 1]

        drop = np.zeros(1, dtype=dtype)[0]
        dd = np.zeros(1, dtype=dtype)[0]
        lenl = 0  # E part (cols < nB)
        lenu = 0  # S part off-diagonals (cols != i, cols >= nB)

        iw[i] = nB  # Marker

        # Copy row entries to working arrays
        for k in range(row_start, row_end):
            col = A_indices[k]
            val = A_data[k]

            if col < nB:
                # E part (goes into L)
                iw[col] = lenl
                levl[col] = 0
                iL[lenl] = col
                wL[lenl] = val
                lenl += 1
            elif col != i:
                # S part (off-diagonal)
                iw[col] = lenu
                levu[col] = 0
                iU[lenu] = col
                wU[lenu] = val
                lenu += 1
            else:
                # Diagonal of S
                dd = val

        # Process L entries with heap for level-based fill
        l_heap = [(iL[j], 0) for j in range(lenl)]
        heapq.heapify(l_heap)
        processed = set()

        while l_heap:
            jpiv, lev_jpiv = heapq.heappop(l_heap)
            if jpiv in processed:
                continue
            if lev_jpiv > lfilef:
                levl[jpiv] = -1
                continue
            processed.add(jpiv)

            # Look at U[jpiv, :] to create fill-in
            for ucol, ulev in U_levels[jpiv]:
                new_lev = lev_jpiv + ulev + 1

                if ucol < nB:
                    # Fill in E part
                    if new_lev <= lfilef:
                        if iw[ucol] < 0:
                            iw[ucol] = lenl
                            levl[ucol] = new_lev
                            iL[lenl] = ucol
                            wL[lenl] = np.zeros(1, dtype=dtype)[0]
                            lenl += 1
                            heapq.heappush(l_heap, (ucol, new_lev))
                        elif levl[ucol] > new_lev:
                            levl[ucol] = new_lev
                            heapq.heappush(l_heap, (ucol, new_lev))
                elif ucol != i:
                    # Fill in S part (off-diagonal)
                    if new_lev <= lfils:
                        if iw[ucol] < 0:
                            iw[ucol] = lenu
                            levu[ucol] = new_lev
                            iU[lenu] = ucol
                            wU[lenu] = np.zeros(1, dtype=dtype)[0]
                            lenu += 1
                        elif levu[ucol] > new_lev:
                            levu[ucol] = new_lev

        # Sort L entries by column index
        if lenl > 0:
            order = np.argsort(iL[:lenl])
            iL[:lenl] = iL[:lenl][order]
            wL[:lenl] = wL[:lenl][order]
            levl_temp = np.array([levl[iL[j]] for j in range(lenl)])
            for j in range(lenl):
                iw[iL[j]] = j
                levl[iL[j]] = levl_temp[j]

        # Elimination
        for j in range(lenl):
            jpiv = iL[j]
            if levl[jpiv] < 0 or levl[jpiv] > lfilef:
                continue
            dpiv = wL[j] * D[jpiv]
            wL[j] = dpiv

            for ucol, uval in U_values[jpiv].items():
                jpos = iw[ucol]

                if jpos < 0:
                    drop = drop - uval * dpiv
                    continue

                lxu = -uval * dpiv

                if ucol < nB:
                    wL[jpos] = wL[jpos] + lxu
                elif ucol != i:
                    wU[jpos] = wU[jpos] + lxu
                else:
                    dd = dd + lxu

        if modified:
            dd = dd + drop

        # Reset iw
        iw[i] = -1

        # Filter and store L entries
        final_lenl = 0
        for j in range(lenl):
            col = iL[j]
            if levl[col] >= 0 and levl[col] <= lfilef:
                iL[final_lenl] = col
                wL[final_lenl] = wL[j]
                final_lenl += 1
            iw[col] = -1
            levl[col] = -1
        lenl = final_lenl

        # Filter S entries
        final_lenu = 0
        for j in range(lenu):
            col = iU[j]
            if levu[col] >= 0 and levu[col] <= lfils:
                iU[final_lenu] = col
                wU[final_lenu] = wU[j]
                final_lenu += 1
            iw[col] = -1
            levu[col] = -1
        lenu = final_lenu

        # Store L entries (E * U^{-1} part)
        if lenl > 0:
            if L_ptr + lenl > L_size:
                while L_ptr + lenl > L_size:
                    L_size = int(L_size * expand_fact + 1)
                L_indices = np.resize(L_indices, L_size)
                L_data = np.resize(L_data, L_size)

            order = np.argsort(iL[:lenl])
            L_indices[L_ptr : L_ptr + lenl] = iL[:lenl][order]
            L_data[L_ptr : L_ptr + lenl] = wL[:lenl][order]
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
