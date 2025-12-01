"""
Base class for ILU factorization results.
"""

from typing import Optional, Tuple, Union, Any
import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import LinearOperator


class ILUResult:
    """
    Result of an ILU factorization.

    This class stores the L, D, U factors from ILU factorization, along with
    optional Schur complement parts (E, F, S) when partial factorization is used.

    The factorization is stored as:
        - L: Lower triangular, unit diagonal (diagonal NOT stored, implicit 1s)
        - D: Diagonal of U stored as inverse (1/d_ii)
        - U: Upper triangular, no diagonal (diagonal stored in D)
        - E: Lower-left block (E * U^{-1}) when nB < n
        - F: Upper-right block (L^{-1} * F) when nB < n
        - S: Schur complement when nB < n

    For the block structure:
        A = [B  F]    L = [LB      0  ]    U = [UB   L^{-1}F]
            [E  C]        [EU^{-1} 0  ]        [0    0      ]

    Attributes
    ----------
    L : csr_matrix
        Lower triangular factor (without diagonal).
    D : ndarray
        Diagonal of U as inverse values (1/d_ii).
    U : csr_matrix
        Upper triangular factor (without diagonal).
    E : csr_matrix or None
        Lower-left block E * U^{-1} (None if nB == n).
    F : csr_matrix or None
        Upper-right block L^{-1} * F (None if nB == n).
    S : csr_matrix or None
        Schur complement (None if nB == n).
    n : int
        Matrix dimension.
    nB : int
        Size of the B block (nB == n means full factorization).
    dtype : numpy dtype
        Data type of the factorization.
    """

    def __init__(
        self,
        L: csr_matrix,
        D: npt.NDArray[Any],
        U: csr_matrix,
        E: Optional[csr_matrix] = None,
        F: Optional[csr_matrix] = None,
        S: Optional[csr_matrix] = None,
        n: Optional[int] = None,
        nB: Optional[int] = None,
    ) -> None:
        """
        Initialize ILUResult.

        Parameters
        ----------
        L : csr_matrix
            Lower triangular factor (without diagonal).
        D : ndarray
            Diagonal of U as inverse values (1/d_ii).
        U : csr_matrix
            Upper triangular factor (without diagonal).
        E : csr_matrix, optional
            Lower-left block.
        F : csr_matrix, optional
            Upper-right block.
        S : csr_matrix, optional
            Schur complement.
        n : int, optional
            Matrix dimension (inferred from L if not provided).
        nB : int, optional
            Size of B block (n if not provided).
        """
        self.L = L
        self.D = D
        self.U = U
        self.E = E
        self.F = F
        self.S = S

        if n is None:
            n = L.shape[0]
        self.n = n

        if nB is None:
            nB = n
        self.nB = nB

        self.dtype = L.dtype

    def to_complete(self) -> Tuple[csr_matrix, csr_matrix]:
        """
        Convert to complete L and U matrices with diagonals.

        Returns complete L (with unit diagonal) and U (with diagonal from D).
        Only returns the B block factors (first nB rows/cols).

        Returns
        -------
        L_complete : csr_matrix
            Complete lower triangular matrix (nB x nB) with unit diagonal.
        U_complete : csr_matrix
            Complete upper triangular matrix (nB x nB) with diagonal.
        """
        nB = self.nB

        # Extract B block of L (first nB rows, first nB cols)
        L_B = self.L[:nB, :nB]
        U_B = self.U[:nB, :nB]

        # Add unit diagonal to L
        L_complete = L_B + sparse.eye(nB, format='csr', dtype=self.dtype)

        # Add diagonal to U (D stores inverse: use 1/D)
        D_diag = diags(1.0 / self.D[:nB], 0, format='csr', dtype=self.dtype)
        U_complete = U_B + D_diag

        return L_complete, U_complete

    def solve(self, b: npt.ArrayLike) -> npt.NDArray[Any]:
        """
        Solve the preconditioner system: M * x = b, where M ~ L * D^{-1} * U.

        This solves: L * D^{-1} * U * x = b
        Which is: x = U^{-1} * D * L^{-1} * b

        For partial factorization (nB < n), only solves the B block.

        Parameters
        ----------
        b : ndarray
            Right-hand side vector.

        Returns
        -------
        x : ndarray
            Solution vector.
        """
        b_arr = np.asarray(b).flatten()

        if self.nB == self.n:
            # Full factorization: solve L * y = b, then U * x = D * y
            return self._solve_full(b_arr)
        else:
            # Partial factorization: only solve B block
            return self._solve_partial(b_arr)

    def _solve_full(self, b: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Solve with full factorization."""
        n = self.n

        # Forward solve: L * z = b (L has unit diagonal)
        z = self._forward_solve(b)

        # Backward solve: U_complete * x = z (includes diagonal from D)
        x = self._backward_solve(z)

        return x

    def _solve_partial(self, b: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Solve with partial factorization (B block only)."""
        n = self.n
        nB = self.nB
        x = np.zeros(n, dtype=self.dtype)

        # Only solve the B block part
        b_B = b[:nB]

        # Forward solve: L_B * z = b_B
        z = self._forward_solve_block(b_B, nB)

        # Backward solve: U_B_complete * x_B = z (includes diagonal from D)
        x[:nB] = self._backward_solve_block(z, nB)

        # For the C block, just copy (no factorization there)
        x[nB:] = b[nB:]

        return x

    def _forward_solve(self, b: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Forward solve: L * y = b where L has unit diagonal.
        L is stored without diagonal.
        """
        n = self.n
        y = b.copy()

        L_indptr = self.L.indptr
        L_indices = self.L.indices
        L_data = self.L.data

        for i in range(n):
            # y[i] = b[i] - sum(L[i,j] * y[j] for j < i)
            start = L_indptr[i]
            end = L_indptr[i + 1]
            for k in range(start, end):
                j = L_indices[k]
                y[i] -= L_data[k] * y[j]
            # Unit diagonal, so no division needed

        return y

    def _backward_solve(self, z: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Backward solve: U_complete * x = z where U_complete = U + diag(1/D).
        D stores inverse of diagonal, so diagonal of U_complete is 1/D[i].
        """
        n = self.n
        x = z.copy()

        U_indptr = self.U.indptr
        U_indices = self.U.indices
        U_data = self.U.data

        for i in range(n - 1, -1, -1):
            # x[i] = (z[i] - sum(U[i,j] * x[j] for j > i)) / (1/D[i])
            #      = (z[i] - sum(U[i,j] * x[j] for j > i)) * D[i]
            start = U_indptr[i]
            end = U_indptr[i + 1]
            for k in range(start, end):
                j = U_indices[k]
                x[i] -= U_data[k] * x[j]
            x[i] *= self.D[i]

        return x

    def _forward_solve_block(self, b: npt.NDArray[Any], nB: int) -> npt.NDArray[Any]:
        """Forward solve for B block only."""
        y = b.copy()

        L_indptr = self.L.indptr
        L_indices = self.L.indices
        L_data = self.L.data

        for i in range(nB):
            start = L_indptr[i]
            end = L_indptr[i + 1]
            for k in range(start, end):
                j = L_indices[k]
                if j < nB:
                    y[i] -= L_data[k] * y[j]

        return y

    def _backward_solve_block(self, z: npt.NDArray[Any], nB: int) -> npt.NDArray[Any]:
        """Backward solve for B block only, including diagonal from D."""
        x = z.copy()

        U_indptr = self.U.indptr
        U_indices = self.U.indices
        U_data = self.U.data

        for i in range(nB - 1, -1, -1):
            start = U_indptr[i]
            end = U_indptr[i + 1]
            for k in range(start, end):
                j = U_indices[k]
                if j < nB:
                    x[i] -= U_data[k] * x[j]
            x[i] *= self.D[i]

        return x

    def __call__(self, b: npt.ArrayLike) -> npt.NDArray[Any]:
        """
        Make the ILUResult callable for use as a preconditioner.

        Parameters
        ----------
        b : ndarray
            Right-hand side vector.

        Returns
        -------
        x : ndarray
            Solution to preconditioner system.
        """
        return self.solve(b)

    def to_linear_operator(self) -> LinearOperator:
        """
        Convert to a scipy LinearOperator for the preconditioner solve.

        Returns
        -------
        M : LinearOperator
            Linear operator that applies the preconditioner solve.
        """
        n = self.n
        dtype = self.dtype

        return LinearOperator(
            shape=(n, n),
            matvec=self.solve,
            dtype=dtype
        )

    @property
    def nnz(self) -> int:
        """Total number of nonzeros in L, U, and S."""
        total = self.L.nnz + self.U.nnz
        if self.S is not None:
            total += self.S.nnz
        return total

    def __repr__(self) -> str:
        s = f"ILUResult(n={self.n}, nB={self.nB}, "
        s += f"nnz(L)={self.L.nnz}, nnz(U)={self.U.nnz}"
        if self.S is not None:
            s += f", nnz(S)={self.S.nnz}"
        s += f", dtype={self.dtype})"
        return s
