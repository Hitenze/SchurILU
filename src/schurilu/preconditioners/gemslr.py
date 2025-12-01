"""
GeMSLR - General Multilevel Schur Low-Rank Preconditioner.

This module implements the GeMSLR preconditioner which combines:
1. Multilevel domain decomposition
2. ILU factorization on interior blocks
3. Low-rank correction for Schur complement approximation
"""

from typing import Optional, List, Tuple, Callable, Any, Dict, Union
import warnings
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, issparse, spmatrix
from scipy.sparse.linalg import LinearOperator

from schurilu.preconditioners.ilut import ilut
from schurilu.preconditioners.iluk import iluk
from schurilu.reordering import multilevel_partition


def arnoldi(
    matvec: Callable[[npt.NDArray[Any]], npt.NDArray[Any]],
    n: int,
    neig: int = 5,
    neig_keep: Optional[int] = None,
    tol: float = 1e-12,
    maxiter: int = 600,
    orthtol: float = 1e-16,
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], int, int]:
    """
    Arnoldi iteration with deflation to find dominant eigenvalues.

    Parameters
    ----------
    matvec : callable
        Function that computes matrix-vector product A*v.
    n : int
        Size of the matrix.
    neig : int, optional
        Target number of eigenvalues to find. Default is 5.
    neig_keep : int, optional
        Number of eigenvalues to keep. Default is same as neig.
    tol : float, optional
        Convergence tolerance for eigenvalues. Default is 1e-12.
    maxiter : int, optional
        Maximum number of Arnoldi iterations. Default is 600.
    orthtol : float, optional
        Orthogonalization tolerance. Default is 1e-16.

    Returns
    -------
    V : ndarray
        Matrix of Ritz vectors (n x m), orthonormal columns.
    H : ndarray
        Upper triangular Schur form (m x m).
    m : int
        Number of converged eigenvalues.
    converged : int
        Total number of iterations performed.
    """
    from scipy.linalg import schur

    if neig_keep is None:
        neig_keep = neig

    neig_keep = min(neig_keep, neig)

    if n == 0:
        return np.zeros((0, 0)), np.zeros((0, 0)), 0, 0

    # Limit iterations to matrix size
    maxiter = min(maxiter, n)
    neig = min(neig, n)
    neig_keep = min(neig_keep, n)

    # Initialize
    m = min(neig + 10, maxiter, n)  # Krylov subspace dimension
    V = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))

    # Random starting vector
    v0 = np.random.randn(n)
    V[:, 0] = v0 / np.linalg.norm(v0)

    alpha = 1.0 / np.sqrt(2.0)  # Re-orthogonalization threshold
    tits = 0
    check_interval = 10

    for i in range(m):
        tits += 1

        # Compute A*v
        V[:, i + 1] = matvec(V[:, i])

        # Store norm before orthogonalization
        normv = np.linalg.norm(V[:, i + 1])

        # Modified Gram-Schmidt
        for j in range(i + 1):
            h = np.dot(V[:, j], V[:, i + 1])
            H[j, i] = h
            V[:, i + 1] = V[:, i + 1] - h * V[:, j]

        # Re-orthogonalization if needed
        t = np.linalg.norm(V[:, i + 1])
        while t >= orthtol and t < alpha * normv:
            normv = t
            for j in range(i + 1):
                h = np.dot(V[:, j], V[:, i + 1])
                H[j, i] += h
                V[:, i + 1] = V[:, i + 1] - h * V[:, j]
            t = np.linalg.norm(V[:, i + 1])

        H[i + 1, i] = t

        # Check for breakdown
        if t < orthtol:
            # Lucky breakdown - all eigenvalues are accurate
            m = i + 1
            break

        V[:, i + 1] = V[:, i + 1] / t

        # Periodically check for convergence
        if (i + 1) % check_interval == 0 or i == m - 1:
            m1 = i + 1
            Hc = H[:m1, :m1]
            Vc = V[:, :m1]

            # Schur decomposition with sorting by eigenvalue magnitude (descending)
            # sort='ouc' = outside unit circle = largest magnitude first
            try:
                # Sort by magnitude descending
                def sort_func(x):
                    return np.abs(x)

                # schur with sort returns (T, Z, sdim) - 3 values
                H1, X1, sdim = schur(Hc, output="real", sort=sort_func)
                eigenvalues = np.diag(H1)
                vals = np.abs(eigenvalues)

                # Compute residuals using Schur vectors
                vres = np.abs(H[m1, m1 - 1]) * np.abs(X1[m1 - 1, :])

                # Count converged eigenvalues (top neig_keep)
                converged_count = np.sum(vres[:neig] < tol)

                if converged_count >= neig:
                    m_out = min(neig_keep, m1)
                    V_out = Vc @ X1[:, :m_out]
                    H_out = H1[:m_out, :m_out]
                    return V_out, H_out, m_out, tits

            except Exception:
                pass

    # End of iterations - extract what we have
    m1 = min(tits, m)
    if m1 == 0:
        return np.zeros((n, 0)), np.zeros((0, 0)), 0, tits

    Hc = H[:m1, :m1]
    Vc = V[:, :m1]

    try:
        # Sort by magnitude descending
        def sort_func(x):
            return np.abs(x)

        # schur with sort returns (T, Z, sdim) - 3 values
        H1, X1, sdim = schur(Hc, output="real", sort=sort_func)

        m_out = min(neig_keep, m1)
        V_out = Vc @ X1[:, :m_out]
        H_out = H1[:m_out, :m_out]

        return V_out, H_out, m_out, tits

    except Exception:
        return np.zeros((n, 0)), np.zeros((0, 0)), 0, tits


class GeMSLR:
    """
    GeMSLR - General Multilevel Schur Low-Rank Preconditioner.

    Combines multilevel domain decomposition, ILU factorization,
    and low-rank correction for the Schur complement.

    Parameters
    ----------
    A : sparse matrix
        Input matrix.
    p : ndarray, optional
        User-provided permutation. If None, spectral partitioning is used.
    lev_ptr : ndarray, optional
        Level pointers (required if p is provided).
    nlev : int, optional
        Number of levels (used if p is None). Default is 2.
    k : int, optional
        Number of partitions per level (used if p is None). Default is 4.
    minsep : int, optional
        Minimum separator size (used if p is None). Default is 16.
    ilu_type : str, optional
        ILU type: 'ilut' or 'iluk'. Default is 'ilut'.
    droptol : float, optional
        Drop tolerance for ILU. Default is 1e-2.
    lfil : int, optional
        Fill limit for ILU. Default is 100.
    level_k : int, optional
        Level of fill for ILUK. Default is 1.
    rank_k : int, optional
        Target rank for low-rank correction. Default is 5.
    theta : float, optional
        Spectrum shift parameter. Default is 0.0.
    arnoldi_tol : float, optional
        Tolerance for Arnoldi eigenvalue computation. Default is 1e-12.
    arnoldi_maxiter : int, optional
        Maximum Arnoldi iterations. Default is 600.

    Attributes
    ----------
    n : int
        Matrix dimension.
    nlev : int
        Number of levels.
    p : ndarray
        Permutation array.
    lev_ptr : ndarray
        Level pointers.
    """

    def __init__(
        self,
        A: Union[spmatrix, npt.ArrayLike],
        p: Optional[npt.NDArray[np.int64]] = None,
        lev_ptr: Optional[npt.NDArray[np.int64]] = None,
        nlev: int = 2,
        k: int = 4,
        minsep: int = 16,
        ilu_type: str = "ilut",
        droptol: float = 1e-2,
        lfil: int = 100,
        level_k: int = 1,
        rank_k: int = 5,
        theta: float = 0.0,
        arnoldi_tol: float = 1e-12,
        arnoldi_maxiter: int = 600,
    ) -> None:
        if not issparse(A):
            A = csr_matrix(A)

        self.A_orig = A.tocsr()
        self.n = A.shape[0]
        self.dtype = A.dtype

        # Store parameters
        self.ilu_type = ilu_type
        self.droptol = droptol
        self.lfil = lfil
        self.level_k = level_k
        self.rank_k = rank_k
        self.theta = theta
        self.arnoldi_tol = arnoldi_tol
        self.arnoldi_maxiter = arnoldi_maxiter

        # Get or compute permutation
        if p is not None:
            if lev_ptr is None:
                raise ValueError("lev_ptr must be provided when p is given")
            self.p = np.asarray(p, dtype=np.int64)
            self.lev_ptr = np.asarray(lev_ptr, dtype=np.int64)
            self.nlev = len(lev_ptr) - 1
        else:
            self.p, self.lev_ptr, self.nlev = multilevel_partition(
                A, nlev=nlev, k=k, minsep=minsep
            )

        # Inverse permutation
        self.p_inv = np.zeros(self.n, dtype=np.int64)
        self.p_inv[self.p] = np.arange(self.n)

        # Permuted matrix
        self.A = self.A_orig[self.p, :][:, self.p].tocsr()

        # Build preconditioner
        self._setup()

    def _setup(self) -> None:
        """Build the multilevel ILU and low-rank corrections."""
        n = self.n
        nlev = self.nlev
        lev_ptr = self.lev_ptr
        A = self.A

        # Store level data
        self.levels: List[Dict[str, Any]] = []

        # Phase 1: ILU factorization on each level
        for levi in range(nlev - 1):
            n_s = lev_ptr[levi]
            n_e = lev_ptr[levi + 1]

            level = {}

            # Extract blocks
            # A_level = A[n_s:n, n_s:n]  # Full remaining matrix
            B = A[n_s:n_e, n_s:n_e]  # Interior block
            F = A[n_s:n_e, n_e:n]  # Upper-right coupling
            E = A[n_e:n, n_s:n_e]  # Lower-left coupling
            C = A[n_e:n, n_e:n]  # Schur complement region

            level["B"] = B.tocsr()
            level["F"] = F.tocsr()
            level["E"] = E.tocsr()
            level["C"] = C.tocsr()
            level["n_s"] = n_s
            level["n_e"] = n_e
            level["nB"] = n_e - n_s
            level["nC"] = n - n_e

            # ILU of B
            if self.ilu_type == "iluk":
                level["ilu_B"] = iluk(B, lfil=self.level_k)
            else:
                level["ilu_B"] = ilut(B, droptol=self.droptol, lfil=self.lfil)

            self.levels.append(level)

        # Last level: ILU of remaining block
        n_s = lev_ptr[nlev - 1]
        C_last = A[n_s:n, n_s:n]

        level_last = {}
        level_last["C"] = C_last.tocsr()
        level_last["n_s"] = n_s
        level_last["nC"] = n - n_s

        if self.ilu_type == "iluk":
            level_last["ilu_C"] = iluk(C_last, lfil=self.level_k)
        else:
            level_last["ilu_C"] = ilut(C_last, droptol=self.droptol, lfil=self.lfil)

        self.levels.append(level_last)

        # Phase 2: Build low-rank corrections (bottom-up)
        if self.rank_k > 0:
            for levi in range(nlev - 2, -1, -1):
                self._build_low_rank(levi)
        else:
            # No low-rank correction
            for levi in range(nlev - 1):
                level = self.levels[levi]
                nC = level["nC"]
                level["Z"] = np.zeros((nC, 0))
                level["G"] = np.zeros((0, 0))

    def _build_low_rank(self, levi: int) -> None:
        """Build low-rank correction for level levi."""
        level = self.levels[levi]
        nC = level["nC"]

        if nC == 0 or self.rank_k == 0:
            level["Z"] = np.zeros((nC, 0))
            level["G"] = np.zeros((0, 0))
            return

        rank_k = min(self.rank_k, nC)

        # Define the SCinv operator: x -> (I - S*C^{-1}) * x
        # Where S = C - E * B^{-1} * F is the exact Schur complement
        def SCinv(x):
            # y = x
            y = x.copy()

            # y1 = C^{-1} * x (solve with C using recursive solve)
            y1 = self._solve_C(x, levi + 1)

            # y = y - C * y1
            y = y - level["C"] @ y1

            # y1 = F * y1
            y1 = level["F"] @ y1

            # y1 = B^{-1} * y1
            y1 = level["ilu_B"].solve(y1)

            # y = y + E * y1
            y = y + level["E"] @ y1

            return y

        # Run Arnoldi to find eigenvalues of SCinv
        V, H, m, tits = arnoldi(
            SCinv,
            nC,
            neig=rank_k,
            neig_keep=rank_k,
            tol=self.arnoldi_tol,
            maxiter=self.arnoldi_maxiter,
        )

        if m == 0:
            level["Z"] = np.zeros((nC, 0))
            level["G"] = np.zeros((0, 0))
            return

        # Build low-rank correction
        # G = (I - H)^{-1} - (1-theta)^{-1} * I
        I_m = np.eye(m)
        try:
            G = np.linalg.inv(I_m - H) - (1.0 / (1.0 - self.theta)) * I_m
        except np.linalg.LinAlgError:
            warnings.warn(
                "Low-rank correction matrix (I - H) is singular; "
                "falling back to zero correction.",
                RuntimeWarning,
                stacklevel=2,
            )
            G = np.zeros((m, m))

        level["Z"] = V  # Ritz vectors
        level["G"] = G  # Low-rank coefficient matrix

    def _solve_C(self, x: npt.NDArray[Any], levi: int) -> npt.NDArray[Any]:
        """
        Solve with C block at level levi (recursive).

        This is used during low-rank construction to approximate C^{-1}.
        """
        if levi >= self.nlev:
            return x

        level = self.levels[levi]

        if levi == self.nlev - 1:
            # Last level: direct ILU solve
            return level["ilu_C"].solve(x)
        else:
            # Recursive solve
            return self._solve_level(x, levi)

    def _solve_level(self, x: npt.NDArray[Any], levi: int) -> npt.NDArray[Any]:
        """
        Apply preconditioner solve at level levi.

        Implements the recursive GeMSLR solve:
        1. Split x into (xB, xC)
        2. zB = ILU(B)^{-1} * xB
        3. zC = xC - E * zB
        4. zC = zC + Z * G * Z' * zC  (low-rank correction)
        5. yC = solve_level(zC, levi+1)  (recursive)
        6. yB = zB - ILU(B)^{-1} * F * yC
        7. return (yB, yC)
        """
        if levi >= self.nlev:
            return x

        level = self.levels[levi]

        if levi == self.nlev - 1:
            # Last level: direct ILU solve
            return level["ilu_C"].solve(x)

        nB = level["nB"]

        # Split input
        xB = x[:nB]
        xC = x[nB:]

        # Solve with B
        zB = level["ilu_B"].solve(xB)

        # Update C part
        zC = xC - level["E"] @ zB

        # Apply low-rank correction
        if level["Z"].shape[1] > 0:
            zC = zC + level["Z"] @ (level["G"] @ (level["Z"].T @ zC))

        # Recursive solve with C
        yC = self._solve_level(zC, levi + 1)

        # Back-substitute
        yB = zB - level["ilu_B"].solve(level["F"] @ yC)

        return np.concatenate([yB, yC])

    def solve(self, b: npt.ArrayLike) -> npt.NDArray[Any]:
        """
        Apply preconditioner: y = M^{-1} * b.

        Parameters
        ----------
        b : ndarray
            Input vector.

        Returns
        -------
        y : ndarray
            Preconditioned vector.
        """
        b = np.asarray(b).flatten()

        if len(b) != self.n:
            raise ValueError(f"Input size {len(b)} != matrix size {self.n}")

        # Apply permutation
        b_perm = b[self.p]

        # Solve in permuted space
        y_perm = self._solve_level(b_perm, 0)

        # Apply inverse permutation
        y = y_perm[self.p_inv]

        return y

    def __call__(self, b: npt.ArrayLike) -> npt.NDArray[Any]:
        """Make preconditioner callable for FGMRES compatibility."""
        return self.solve(b)

    def to_linear_operator(self) -> LinearOperator:
        """Convert to scipy LinearOperator."""
        return LinearOperator(
            shape=(self.n, self.n),
            matvec=self.solve,
            dtype=self.dtype
        )

    @property
    def nnz_ilu(self) -> int:
        """Total nonzeros in ILU factors."""
        total = 0
        for levi in range(self.nlev - 1):
            level = self.levels[levi]
            total += level["ilu_B"].nnz
        # Last level
        total += self.levels[-1]["ilu_C"].nnz
        return total

    @property
    def nnz_lowrank(self) -> int:
        """Total nonzeros in low-rank corrections."""
        total = 0
        for levi in range(self.nlev - 1):
            level = self.levels[levi]
            if "Z" in level:
                total += level["Z"].size + level["G"].size
        return total

    @property
    def nnz(self) -> int:
        """Total nonzeros in preconditioner."""
        return self.nnz_ilu + self.nnz_lowrank

    def fill_factor(self) -> float:
        """Compute fill factor relative to original matrix."""
        return self.nnz / self.A_orig.nnz

    def __repr__(self) -> str:
        s = f"GeMSLR(n={self.n}, nlev={self.nlev}"
        s += f", nnz_ilu={self.nnz_ilu}, nnz_lr={self.nnz_lowrank}"
        s += f", fill={self.fill_factor():.2f})"
        return s
