# Implementation: Binary Multi-Hard Tri-Factorization (scikit-learn style)
# with Gaussian/Poisson losses
import numpy as np
from typing import Optional, Tuple, Dict, List, Any, Callable
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array
from src.utils import get_logger
import time

logger = get_logger(__name__)


def _mode_ignore_minus1(vec: np.ndarray, K: int) -> int:
    r"""
    Compute the mode of a vector, ignoring -1 values.

    Parameters
    ----------
    vec : (N,) int array
        Vector to compute the mode of.
    K : int
        Number of clusters.

    Returns
    -------
    mode : int
        Most frequent non-negative value in vec, or -1 if all values are negative.
    """
    v = vec[vec >= 0]
    if v.size == 0:
        return -1
    counts = np.bincount(v, minlength=K)
    return int(np.argmax(counts))  # ties → lowest index


# ----------------------------
# Losses
# ----------------------------
def gaussian_loss(
    X: np.ndarray, Xhat: np.ndarray, mask: np.ndarray = None
) -> float:
    if mask is None:
        diff = X - Xhat
    else:
        diff = (X - Xhat)[mask]
    return float(np.sum(diff * diff))


def poisson_nll(
    X: np.ndarray, Xhat: np.ndarray, mask: np.ndarray = None, eps=1e-9
) -> float:
    # NLL(μ; x) = μ - x*log μ (+ const)
    if mask is None:
        x = X
        mu = np.maximum(Xhat, eps)
    else:
        x = X[mask]
        mu = np.maximum(Xhat[mask], eps)
    return float(np.sum(mu - x * np.log(mu)))


def model_loss(
    X: np.ndarray, Xhat: np.ndarray, loss_name, mask: np.ndarray = None
) -> float:
    if loss_name == "gaussian":
        return gaussian_loss(X, Xhat, mask)
    elif loss_name == "poisson":
        return poisson_nll(X, Xhat, mask)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


def elbow_cutoff_by_gap(values: np.ndarray, min_floor: int = 1):
    vals = np.asarray(values, float)
    vals = vals[np.isfinite(vals)]
    vals = np.sort(vals)
    if vals.size == 0:
        return min_floor
    diffs = np.diff(vals) / np.maximum(vals[:-1], 1e-12)  # relative jump
    if diffs.size == 0:
        return max(min_floor, int(vals[0]))
    k = int(np.argmax(diffs))  # index before the largest jump
    cutoff = vals[k]  # everything >= this is "kept"
    return int(max(min_floor, np.floor(cutoff)))


def allowed_mask_from_stats(
    stats: pd.DataFrame,
    rule: str = "delta_then_size",
    size_quantile: float = 0.50,
    delta_quantile: float = 0.50,
    min_rows: int = 1,
    min_cols: int = 1,
) -> np.ndarray:
    R = int(stats["r"].max()) + 1
    C = int(stats["c"].max()) + 1
    allow = np.zeros((R, C), dtype=bool)

    if rule == "size_gap":
        th = elbow_cutoff_by_gap(
            stats["n_cells"].values, min_floor=min_rows * min_cols
        )
        keep = stats["n_cells"] >= th

    elif rule == "effect_gap":
        th = elbow_cutoff_by_gap(stats["effect_score"].values, min_floor=1)
        keep = stats["effect_score"] >= th

    elif rule == "delta_then_size":
        dth = np.nanquantile(stats["delta_loss"].values, delta_quantile)
        sth = np.nanquantile(stats["n_cells"].values, size_quantile)
        keep = (
            (stats["delta_loss"] > 0)
            & (stats["delta_loss"] >= dth)
            & (stats["n_cells"] >= sth)
        )

    elif rule == "quantiles":
        sth = np.nanquantile(stats["n_cells"].values, size_quantile)
        keep = stats["n_cells"] >= sth

    else:
        raise ValueError(f"unknown rule: {rule}")

    # also enforce row/col minimums if desired
    keep &= (stats["n_rows"] >= min_rows) & (stats["n_cols"] >= min_cols)

    for _, row in stats[keep].iterrows():
        allow[int(row["r"]), int(row["c"])] = True
    return allow


class BinaryTriFactorizationEstimator(BaseEstimator, ClusterMixin):
    """
    Binary (multi-hard) tri-factorization:
        X ≈ U B V^T
    where:
      - U ∈ {0,1}^{I×R}  (rows → row-clusters; multiple 1s allowed per row)
      - V ∈ {0,1}^{J×C}  (cols → col-clusters; multiple 1s allowed per column)
      - B ∈ R^{R×C}      (block interaction/means; ≥0 recommended for Poisson)

    Loss:
      - "gaussian": 0.5 ||X - U B V^T||_F^2 + β (||U||_0 + ||V||_0)
      - "poisson":  (up to constants)  Σ_ij [ μ_ij - X_ij log μ_ij ]  + β (||U||_0 + ||V||_0)
                    where μ = U B V^T

    Overlap control:
      - k_row, k_col: max #selected clusters per row/col. Use None for *data-driven* stopping.
      - β penalizes each active bit (sparser U/V for larger β).
    """

    def __init__(
        self,
        n_row_clusters: int,
        n_col_clusters: int,
        k_row: Optional[int] = None,
        k_col: Optional[int] = None,
        loss: str = "gaussian",
        alpha: float = 1e-3,
        beta: float = 0.0,
        max_iter: int = 30,
        tol: float = 1e-4,
        random_state: int = 0,
        block_l1: float = 0.0,
        b_inner: int = 15,
        patience: int = 2,
    ):
        """
        Parameters
        ----------
        n_row_clusters : int
            Number of row clusters
        n_col_clusters : int
            Number of column clusters
        k_row : int or None, default=None
            Max active clusters per row. None for data-driven stopping.
        k_col : int or None, default=None
            Max active clusters per column. None for data-driven stopping.
        loss : str, default="gaussian"
            Loss function: "gaussian" or "poisson"
        alpha : float, default=1e-3
            Ridge regularization (gaussian) / damping (poisson)
        beta : float, default=0.0
            Per-bit penalty on active memberships
        max_iter : int, default=30
            Maximum number of iterations
        tol : float, default=1e-4
            Tolerance for convergence
        random_state : int or None, default=0
            Random seed for reproducibility
        block_l1 : float, default=0.0
            L1 regularization on B matrix (0 = off)
        b_inner : int, default=15
            Inner prox steps for B when block_l1 > 0
        patience : int, default=2
            Number of iterations to wait for improvement before early stopping
        """
        self.n_row_clusters = n_row_clusters
        self.n_col_clusters = n_col_clusters
        self.k_row = k_row
        self.k_col = k_col
        self.loss = loss
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.block_l1 = block_l1
        self.b_inner = b_inner
        self.patience = patience

        # Learned attributes (filled after fit)
        self.U_ = None  # (I,R) binary
        self.V_ = None  # (J,C) binary
        self.B_ = None  # (R,C) real (≥0 for Poisson updates)
        self.Xhat_ = None  # (I,J) reconstruction
        self.history_flag = False
        self.loss_history_ = None  # list of objective values per outer iter

    # -------- Utilities --------
    def _rng(self):
        """Random generator with the configured seed."""
        return np.random.default_rng(self.random_state)

    def _init_binary(self, shape, k: Optional[int], rng) -> np.ndarray:
        """
        Initialize a binary membership matrix with ≥1 active bit per row.
        If k is None: Bernoulli(small p). Else: exactly k ones per row.
        """
        I, R = shape
        Z = np.zeros((I, R), dtype=np.int8)
        if k is None:
            # Bernoulli probability: here fixed to 0.1 (matches your original code behavior)
            p = min(0.1, max(1 / R, 0.1))  # simplifies to 0.1 for most R
            Z = (rng.random((I, R)) < p).astype(np.int8)
        else:
            kk = min(k, R)
            for i in range(I):
                idx = rng.choice(R, size=kk, replace=False)
                Z[i, idx] = 1
        # Guarantee at least one active bit per row
        empty = Z.sum(axis=1) == 0
        if np.any(empty):
            Z[empty, rng.integers(0, R, size=empty.sum())] = 1
        return Z

    # # -------- B updates --------
    def _update_B_gaussian(self, X, U, V) -> np.ndarray:
        """
        Gaussian B-update.

        Case A (default): separable ridge on UB and BVᵀ  [fast closed form]
            (UᵀU + αI) B (VᵀV + αI) = Uᵀ X V
            solved via two Cholesky systems (no explicit inverses).

        Case B (when block_l1 > 0): ridge-on-B + L1  [ISTA]
            minimize 0.5||X - U B Vᵀ||_F^2 + (α/2)||B||_F^2 + λ||B||_1
            with step size from a Lipschitz bound.
        """
        import logging

        R, C = self.n_row_clusters, self.n_col_clusters

        # Ensure float64 for stable linear algebra
        X = np.asarray(X, dtype=np.float64, order="C")
        U = np.asarray(U, dtype=np.float64, order="C")
        V = np.asarray(V, dtype=np.float64, order="C")

        # ---- Case B: L1 on B -> proximal gradient (ISTA) ----
        if self.block_l1 > 0.0:
            # warm start
            B = (
                np.zeros((R, C), dtype=np.float64)
                if (self.B_ is None)
                else self.B_.copy().astype(np.float64)
            )

            UtU = U.T @ U
            VtV = V.T @ V
            # Lipschitz bound for ∇f(B) = Uᵀ(UBVᵀ - X)V + αB
            L = float(
                np.linalg.norm(UtU, 2) * np.linalg.norm(VtV, 2) + self.alpha
            )
            eta = 1.0 / max(L, 1e-12)

            lam = float(self.block_l1)
            a = float(self.alpha)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Bstep_ISTA start L={L:.3e} eta={eta:.3e} B_L1={np.sum(np.abs(B)):.6e}"
                )

            iters = max(1, int(self.b_inner))
            for t in range(iters):
                # Gradient step
                E = (U @ B) @ V.T - X  # residual in data space
                G = U.T @ E @ V + a * B  # gradient in B-space
                B = B - eta * G
                # Soft-thresholding
                thr = eta * lam
                B = np.sign(B) * np.maximum(np.abs(B) - thr, 0.0)

                if logger.isEnabledFor(logging.DEBUG) and (t == iters - 1):
                    gnorm = float(np.linalg.norm(G))
                    logger.debug(
                        f"Bstep_ISTA end iters={iters} gradB_frob={gnorm:.6e} "
                        f"B_L1={np.sum(np.abs(B)):.6e} B_frob={np.linalg.norm(B):.6e}"
                    )

            return B

        # ---- Case A: separable ridge (default fast path) ----
        UtU = U.T @ U
        VtV = V.T @ V

        # Add αI on the diagonals (SPD even if U/V are rank-deficient when α>0)
        UtU = UtU.copy()
        VtV = VtV.copy()
        UtU.flat[:: R + 1] += self.alpha
        VtV.flat[:: C + 1] += self.alpha

        # Optional diagnostics: condition numbers after adding alpha
        if logger.isEnabledFor(logging.DEBUG):
            try:
                # eigvalsh is cheaper and stable for SPD
                wU = np.linalg.eigvalsh(UtU)
                wV = np.linalg.eigvalsh(VtV)
                cond_U = float(wU.max() / max(wU.min(), 1e-18))
                cond_V = float(wV.max() / max(wV.min(), 1e-18))
            except Exception:
                cond_U = float("inf")
                cond_V = float("inf")
            logger.debug(f"Bstep cond_UtU={cond_U:.3e} cond_VtV={cond_V:.3e}")

        # Right-hand side
        M = U.T @ X @ V

        # Solve (UᵀU+αI) Y = M  via Cholesky (with jitter tracking)
        def chol_solve_with_jitter(A, Bmat):
            """Solve A X = Bmat with Cholesky; add tiny jitter if needed. Returns (X, jitter_total)."""
            jitter = 1e-12
            total = 0.0
            A_work = A  # operate in-place on the provided matrix
            for _ in range(3):
                try:
                    L = np.linalg.cholesky(A_work)
                    Y = np.linalg.solve(L, Bmat)
                    Xsol = np.linalg.solve(L.T, Y)
                    return Xsol, total
                except np.linalg.LinAlgError:
                    diag_incr = jitter
                    A_work.flat[:: A_work.shape[0] + 1] += diag_incr
                    total += diag_incr
                    jitter *= 10.0
            # Final fallback (should be rare)
            Xsol = np.linalg.solve(A_work, Bmat)
            return Xsol, total

        # Left solve: (UᵀU+αI)^{-1} M
        Y, jU = chol_solve_with_jitter(UtU, M)
        # Right solve: B = Y (VᵀV+αI)^{-1}  via solving transposed system
        BT, jV = chol_solve_with_jitter(VtV, Y.T)  # solves (VᵀV+αI) BT = Yᵀ
        B = BT.T

        if logger.isEnabledFor(logging.DEBUG):
            if (jU > 0) or (jV > 0):
                logger.debug(
                    f"Bstep cholesky_jitter diag_increase UtU={jU:.1e} VtV={jV:.1e}"
                )
            # Gradient norm at solution: ∇_B = Uᵀ(U B Vᵀ - X)V + αB
            E = (U @ B) @ V.T - X
            G = U.T @ E @ V + self.alpha * B
            logger.debug(
                f"Bstep closed_form gradB_frob={np.linalg.norm(G):.6e} "
                f"B_frob={np.linalg.norm(B):.6e}"
            )

        return B

    def _update_B_poisson(self, X, U, V, B, n_inner: int = 15) -> np.ndarray:
        r"""
        Poisson/KL update of B with fixed U, V (B ≥ 0).

        Objective (generalized KL divergence with L2/L1 on B):
            minimize_B   D_KL(X || U B Vᵀ)  +  (α/2) ||B||_F^2  +  λ ||B||_1        (★)

        where the generalized KL term is
            D_KL(X || M) = ⟨ M, 1 ⟩ - ⟨ X, log M ⟩ + const,   with M = U B Vᵀ, M>0.

        Gradients of the KL part w.r.t. B:
            ∂/∂B ⟨ M, 1 ⟩         = Uᵀ 1 V
            ∂/∂B ⟨ X, log M ⟩     = Uᵀ (X ⊘ M) V                                     (1)

        Multiplicative update (elementwise), obtained from splitting (+) and (−) parts:
            B ← B ⊙   [ Uᵀ (X ⊘ (U B Vᵀ)) V ]  ⊘  [ Uᵀ 1 V  +  α  +  λ ]            (2)

        Notes:
        • α ≥ 0 is L2 (ridge) on B, incorporated additively in the denominator.
        • λ ≥ 0 (self.block_l1) is the nonnegative-L1 shrinkage, also added to the denominator.
        • ⊙ and ⊘ are Hadamard (elementwise) product and division.
        • Shapes: X∈ℝ^{m×n}, U∈ℝ^{m×k}, V∈ℝ^{n×ℓ}, B∈ℝ^{k×ℓ}; Uᵀ 1 V matches B.
        """
        # small floors to keep M = U B Vᵀ and B strictly positive (needed for KL)
        epsM = 1e-9
        epsB = 1e-12

        # ensure B > 0 to start
        B = np.maximum(B, epsB)

        ones = np.ones_like(X, dtype=X.dtype)

        for _ in range(n_inner):
            # M = U B Vᵀ  (Poisson mean / KL "model" matrix)
            M = (U @ B) @ V.T
            M = np.maximum(M, epsM)  # avoid divide-by-zero and log(0)

            # Numerator in (2): Uᵀ ( X ⊘ M ) V     [from the '−⟨X, log M⟩' term in (1)]
            numer = U.T @ (X / M) @ V

            # Denominator in (2):
            #   Uᵀ 1 V      [from the '+⟨M, 1⟩' term in (1)]
            # + α           [ridge on B]
            # + λ           [L1 shrinkage on nonnegative B]
            denom = (U.T @ ones @ V) + self.alpha + self.block_l1

            # Elementwise multiplicative step (2)
            B *= numer / np.maximum(denom, epsB)

            # Re-enforce positivity (keeps KL well-defined and stable)
            B = np.maximum(B, epsB)

        return B

    def _log_membership_histogram(
        self, M: np.ndarray, name: str, top_bins: int = 10
    ) -> None:
        """
        Log a compact histogram of per-entity membership counts for a binary assignment matrix M.
        """
        if M.size == 0:
            logger.info(f"{name}: empty membership matrix")
            return

        # Each row of M corresponds to one entity (row or column); sum across clusters
        counts = np.asarray(M.sum(axis=1)).astype(int).ravel()
        n = counts.size
        kmax = int(counts.max(initial=0))

        # Histogram: show bins 0..K, and lump any higher counts into a tail
        K = min(kmax, int(top_bins))
        hist = np.bincount(counts, minlength=K + 1)
        shown = hist[: K + 1].sum()
        tail = n - shown

        # Basic stats
        mean = float(counts.mean()) if n else 0.0
        med = float(np.median(counts)) if n else 0.0
        p90 = float(np.percentile(counts, 90)) if n else 0.0
        p95 = float(np.percentile(counts, 95)) if n else 0.0
        p99 = float(np.percentile(counts, 99)) if n else 0.0

        # Build a compact string for the first K bins (e.g., "0:12  1:345  2:789 ...")
        bins_str = "  ".join(f"{k}:{int(hist[k])}" for k in range(K + 1))
        if tail > 0:
            bins_str += f"  (>{K}:{tail})"

        logger.info(
            f"{name} memberships — n={n}\n"
            f"R:{self.n_row_clusters},C:{self.n_col_clusters}\n"
            f"mean={mean:.2f},median={med:.2f}\n"
            f"p90={p90:.2f},p95={p95:.2f},p99={p99:.2f}\n"
        )
        logger.info(f"  counts: {bins_str}")

    # -------- Toggle scoring (marginal gain) --------
    def _poisson_delta_ll_row(self, x_row, mu_row, g_row):
        """
        Exact Δ log-likelihood for Poisson when toggling a row-bit:
          sum_j [ x log(mu+g) - (mu+g) - (x log mu - mu) ]
        """
        mu_new = np.maximum(mu_row + g_row, 1e-9)
        mu_row = np.maximum(mu_row, 1e-9)
        return (
            x_row * (np.log(mu_new) - np.log(mu_row)) - (mu_new - mu_row)
        ).sum()

    def _poisson_delta_ll_col(self, x_col, mu_col, h_col):
        """Column analogue of the Poisson Δ log-likelihood."""
        mu_new = np.maximum(mu_col + h_col, 1e-9)
        mu_col = np.maximum(mu_col, 1e-9)
        return (
            x_col * (np.log(mu_new) - np.log(mu_col)) - (mu_new - mu_col)
        ).sum()

    # helper to compute objective
    def _objective(self, X, Xhat, U, V):
        if self.loss == "gaussian":
            resid = X - Xhat
            penalty = self.beta * (U.sum() + V.sum())
            # objective used for optimization:
            return 0.5 * np.sum(resid * resid) + penalty
        else:
            MU = np.maximum(Xhat, 1e-9)
            return float((MU - X * np.log(MU)).sum()) + self.beta * (
                U.sum() + V.sum()
            )

    def _rss(self, X, Xhat):
        return float(np.sum((X - Xhat) ** 2))

    def _nan_inf_report(self, name: str, arr: np.ndarray) -> dict:
        return dict(
            name=name,
            has_nan=bool(np.isnan(arr).any()),
            has_inf=bool(np.isinf(arr).any()),
            max_abs=float(np.max(np.abs(arr))) if arr.size else 0.0,
            frob=float(np.linalg.norm(arr)) if arr.size else 0.0,
        )

    # def _cond_spd(self, A):
    #     # A is small (R×R or C×C); add tiny diag to avoid crashes in logging
    #     eps = 1e-12
    #     try:
    #         w = np.linalg.eigvalsh(A + eps * np.eye(A.shape[0]))
    #         w = np.clip(w, eps, None)
    #         return float(w.max() / w.min())
    #     except Exception:
    #         return float("inf")

    # def _grad_B_gaussian(self, X, U, V, B, alpha):
    #     # ∇_B 0.5||X - U B Vᵀ||² + (alpha/2)||B||²  = Uᵀ(U B Vᵀ - X)V + alpha B
    #     E = (U @ B) @ V.T - X
    #     return U.T @ E @ V + alpha * B

    # def _hamming_changes(self, A_new, A_prev):
    #     return int(np.sum(A_new != A_prev))

    # def _describe_scores(self, s):
    #     if s.size == 0:
    #         return dict(max=float("-inf"), p95=float("-inf"), pos_frac=0.0)
    #     maxv = float(np.max(s))
    #     p95 = float(np.percentile(s, 95))
    #     pos_frac = float(np.mean(s > 0))
    #     return dict(max=maxv, p95=p95, pos_frac=pos_frac)

    # -------- Fit --------
    def fit(self, X, y=None):
        """
        Fit the binary tri-factorization model to X (Gaussian or Poisson).
        Adds PVE/RMSE logging each iteration.
        """
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        _ = y
        I, J = X.shape
        R, C = self.n_row_clusters, self.n_col_clusters
        rng = self._rng()

        # init memberships
        U = self._init_binary((I, R), self.k_row, rng)
        V = self._init_binary((J, C), self.k_col, rng)

        # init B
        if self.loss == "gaussian":
            B = self._update_B_gaussian(X, U, V)
        elif self.loss == "poisson":
            B = np.abs(rng.normal(0.5, 0.1, size=(R, C))) + 0.1
            B = self._update_B_poisson(X, U, V, B, n_inner=20)
        else:
            raise ValueError("loss must be 'gaussian' or 'poisson'")

        if self.history_flag:
            self.loss_history_ = []

        # baseline
        Xhat = U @ (B @ V.T)
        loss_prev = self._objective(X, Xhat, U, V)
        prev = loss_prev

        logger.info(
            f"R={R},C={C},k_row={self.k_row},k_col={self.k_col},I={I},J={J}\n"
            f"alpha={self.alpha},beta={self.beta},block_l1={self.block_l1},tol={self.tol},\n"
            f"max_iter={self.max_iter},loss={self.loss}"
        )
        logger.debug(f"Nan/inf report:{self._nan_inf_report('X', X)}")

        # for the optional “stable assignments” stop (no new attribute needed)
        n_iters = 0
        self.patience = 2  # consecutive iters with no membership changes
        for it in range(self.max_iter):
            t0 = time.perf_counter()

            # snapshot (rollback + deltas)
            self._u_forced = self._u_positive = 0
            self._v_forced = self._v_positive = 0
            U_prev, V_prev, B_prev = U.copy(), V.copy(), B.copy()
            Xhat_prev = Xhat.copy()
            loss_baseline = loss_prev

            # --- B-step ---
            tB0 = time.perf_counter()
            if self.loss == "gaussian":
                B = self._update_B_gaussian(X, U, V)
            else:
                B = self._update_B_poisson(X, U, V, B, n_inner=self.b_inner)
            tB1 = time.perf_counter()

            # --- Precompute shared quantities ---
            G = B @ V.T
            H = U @ B
            Xhat = U @ G

            if self.loss == "gaussian":
                G_norm2 = np.einsum("rj,rj->r", G, G)
            else:
                G_norm2 = None

            # --- U-step ---
            tU0 = time.perf_counter()
            for i in rng.permutation(I):
                # for i in range(I):
                if self.loss == "gaussian":
                    self._greedy_update_gaussian(
                        X=X,
                        Xhat=Xhat,
                        memberships=U,
                        idx=i,
                        components=G,
                        comp_norm2=G_norm2,
                        k_limit=self.k_row,
                        beta=self.beta,
                        axis=0,
                    )
                else:
                    self._greedy_update_poisson(
                        X=X,
                        Xhat=Xhat,
                        memberships=U,
                        idx=i,
                        components=G,
                        scorer_fn=self._poisson_delta_ll_row,
                        beta=self.beta,
                        k_limit=self.k_row,
                        axis=0,
                    )
            tU1 = time.perf_counter()

            # refresh after U
            Xhat = U @ B @ V.T
            Rres = X - Xhat
            a = Rres.mean(axis=1, keepdims=True)  # I×1
            b = (Rres - a).mean(axis=0, keepdims=True)  # 1×J
            Xhat = Xhat + a + b

            if self.loss == "gaussian":
                H_norm2 = np.einsum("ic,ic->c", H, H)
            else:
                H_norm2 = None

            # --- V-step ---
            tV0 = time.perf_counter()
            for j in rng.permutation(J):
                if self.loss == "gaussian":
                    self._greedy_update_gaussian(
                        X=X,
                        Xhat=Xhat,
                        memberships=V,
                        idx=j,
                        components=H,
                        comp_norm2=H_norm2,
                        k_limit=self.k_col,
                        beta=self.beta,
                        axis=1,
                    )
                else:
                    self._greedy_update_poisson(
                        X=X,
                        Xhat=Xhat,
                        memberships=V,
                        idx=j,
                        components=H,
                        scorer_fn=self._poisson_delta_ll_col,
                        beta=self.beta,
                        k_limit=self.k_col,
                        axis=1,
                    )
            tV1 = time.perf_counter()

            # membership dynamics
            dU = int((U != U_prev).sum())
            dV = int((V != V_prev).sum())
            row_on = float(U.sum(1).mean())
            col_on = float(V.sum(1).mean())

            # optional stability-based stop
            n_iters = n_iters + 1 if (dU == 0 and dV == 0) else 0
            if n_iters >= self.patience:
                logger.info(
                    f"it:{it:02d},R:{R},C:{C} Early stopping (stable assignments for {n_iters} iters)"
                )
                break

            # finalize this outer iteration
            G = B @ V.T
            Xhat = U @ G
            loss_new = self._objective(X, Xhat, U, V)

            # monotonic guard
            rolled_back = False
            if loss_new > loss_baseline + 1e-8:
                rolled_back = True
                U, V, B = U_prev, V_prev, B_prev
                Xhat = Xhat_prev
                loss = loss_baseline
            else:
                loss = loss_new

            if rolled_back:
                logger.warning(
                    f"it:{it:02d},{R},{C} Rollback (new loss    ={loss_new:.6e} > baseline={loss_baseline:.6e})"
                )

            # diagnostics (Gaussian regs shown; l1 only if enabled)
            rss = self._rss(X, Xhat)
            regB = 0.5 * self.alpha * float(np.sum(B * B))
            l1B = float(np.sum(np.abs(B))) if self.block_l1 > 0 else 0.0

            # PVE / RMSE (safe for z-scored data)
            rmse = float(np.sqrt(rss / (I * J)))
            # PVE with MEAN baseline (matches compute_pve)
            eps = 1e-12
            if self.loss == "gaussian":
                mu = float(np.mean(X))
                L_model = float(np.sum((X - Xhat) ** 2))
                L_base = float(np.sum((X - mu) ** 2))
                pve = 1.0 - (L_model / max(L_base, eps))
            else:  # poisson (mean-rate baseline, constants cancel in ratio)
                mu = float(np.mean(X))

                # Poisson NLL without constants: sum(λ - x*log λ); safe log
                def _pois_nll(x, lam):
                    lam_safe = np.maximum(lam, eps)
                    return float(np.sum(lam_safe - x * np.log(lam_safe)))

                L_model = _pois_nll(X, np.maximum(Xhat, eps))
                L_base = _pois_nll(X, mu)
                pve = 1.0 - (L_model / max(L_base, eps))

            # relative contributions
            tot_loss = (
                rss
                + regB
                + (self.block_l1 * l1B if self.block_l1 > 0 else 0.0)
            )
            frac_rss = rss / tot_loss if tot_loss > 0 else 0.0
            frac_reg = regB / tot_loss if tot_loss > 0 else 0.0
            frac_l1 = (
                (self.block_l1 * l1B) / tot_loss
                if self.block_l1 > 0 and tot_loss > 0
                else 0.0
            )

            if self.history_flag:
                self.loss_history_.append(loss)

            logger.info(
                f"it:{it:02d},R:{R},C:{C},loss={loss:.6e},rss={rss:.6e},PVE={pve:.2%},RMSE={rmse:.3f}\n"
                f"RegB={regB:.3e},L1B={l1B:.3e},Frac_rss={frac_rss:.2%}, Frac_reg={frac_reg:.2%}, Frac_l1={frac_l1:.2%}\n"
                f"dU={dU},dV={dV}\n"
                f"Averag_row_clusters={row_on:.2f},averag_col_clusters={col_on:.2f}\n"
                f"U_forced={self._u_forced},U_pos={self._u_positive}\n"
                f"V_forced={self._v_forced},V_pos={self._v_positive}"
            )
            logger.debug(
                f"iter_total={time.perf_counter()-t0:.3f}s,time B={tB1-tB0:.3f}s,U={tU1-tU0:.3f}s,V={tV1-tV0:.3f}s"
            )

            if it % 10 == 0:
                logger.debug(
                    f"it:{it},{self._nan_inf_report('B', B)},{self._nan_inf_report('Xhat', Xhat)}"
                )

            # relative-improvement early stop
            if np.isfinite(prev):
                rel = (prev - loss) / max(1.0, abs(prev))
                if (prev >= loss) and (rel < self.tol):
                    logger.info(
                        f"it:{it},R:{R},C:{C} Early stopping (rel_impr={rel:.6f} < tol={self.tol})"
                    )
                    break
            prev = loss
            loss_prev = loss

        # save
        self.U_, self.V_, self.B_, self.Xhat_ = (
            U.astype(np.int8),
            V.astype(np.int8),
            B,
            Xhat,
        )
        # --- final diagnostics (last accepted iteration) ---
        rows_forced_pct = 100.0 * (self._u_forced / max(1, I))
        rows_pos_pct = 100.0 * (self._u_positive / max(1, I))
        cols_forced_pct = 100.0 * (self._v_forced / max(1, J))
        cols_pos_pct = 100.0 * (self._v_positive / max(1, J))

        logger.info(
            f"Final pick summary — rows: forced={self._u_forced}/{I},({rows_forced_pct:.2f}%)\n"
            f"Positive={self._u_positive}/{I},({rows_pos_pct:.2f}%)\n"
            f"Cols: forced={self._v_forced}/{J},({cols_forced_pct:.2f}%)\n"
            f"Positive={self._v_positive}/{J},({cols_pos_pct:.2f}%)"
        )

        # After the training loop, before saving/returning
        try:
            self._log_membership_histogram(U, name="stores", top_bins=10)
            self._log_membership_histogram(V, name="items", top_bins=10)
        except Exception as e:
            logger.debug(f"membership histogram logging skipped: {e!r}")

        return self

    # -------- Helpers / API --------
    def _greedy_update_gaussian(
        self,
        X: np.ndarray,
        Xhat: np.ndarray,
        memberships: np.ndarray,  # U (I×R) if axis=0, or V (J×C) if axis=1
        idx: int,
        components: (
            np.ndarray | None
        ),  # G (R×J) if axis=0, or H (I×C) if axis=1
        comp_norm2: np.ndarray,  # ||t_k||^2 per template
        k_limit: int | None,
        beta: float,
        axis: int,  # 0=row update, 1=col update
    ) -> None:
        """
        Greedy Gaussian-membership update for one row (axis=0) or one column (axis=1).
        """
        if axis == 0:
            # -------- ROW UPDATE (U) --------
            i = idx
            base = X[i, :] - Xhat[i, :]
            if memberships[i, :].any():
                base = base + components[memberships[i, :] == 1, :].sum(axis=0)

            memberships[i, :] = 0

            # initial scores s = <base, t_k> - 0.5||t_k||^2 - beta
            s = components @ base
            s = s - 0.5 * comp_norm2 - beta

            smax = float(np.max(s)) if s.size else float("-inf")
            logger.debug(f"U[i={i}] smax={smax:.3e}")

            used = np.zeros(components.shape[0], dtype=bool)

            if k_limit == 1:
                r_star = int(np.argmax(s))
                forced = not (s[r_star] > 0)
                self._u_forced += int(forced)
                self._u_positive += int(not forced)
                memberships[i, r_star] = 1
                Xhat[i, :] = memberships[i, :] @ components
                return

            # k > 1
            T = components @ components.T
            picks = 0
            while (k_limit is None) or (picks < k_limit):
                s[used] = -np.inf
                r_star = int(np.argmax(s))
                if s[r_star] <= 0:
                    break
                memberships[i, r_star] = 1
                used[r_star] = True
                picks += 1
                s -= T[r_star, :]

            if memberships[i, :].sum() == 0:
                r_star = int(np.argmax(s))
                memberships[i, r_star] = 1
                self._u_forced += 1
            else:
                self._u_positive += 1

            Xhat[i, :] = memberships[i, :] @ components

        else:
            # -------- COLUMN UPDATE (V) --------
            j = idx
            base = X[:, j] - Xhat[:, j]
            if memberships[j, :].any():
                base = base + components[:, memberships[j, :] == 1].sum(axis=1)

            memberships[j, :] = 0

            s = components.T @ base
            s = s - 0.5 * comp_norm2 - beta

            smax = float(np.max(s)) if s.size else float("-inf")
            logger.debug(f"V[j={j}] smax={smax:.3e}")

            used = np.zeros(components.shape[1], dtype=bool)

            if k_limit == 1:
                c_star = int(np.argmax(s))
                forced = not (s[c_star] > 0)
                self._v_forced += int(forced)
                self._v_positive += int(not forced)
                memberships[j, c_star] = 1
                Xhat[:, j] = components @ memberships[j, :]
                return

            # k > 1
            T = components.T @ components
            picks = 0
            while (k_limit is None) or (picks < k_limit):
                s[used] = -np.inf
                c_star = int(np.argmax(s))
                if s[c_star] <= 0:
                    break
                memberships[j, c_star] = 1
                used[c_star] = True
                picks += 1
                s -= T[c_star, :]

            if memberships[j, :].sum() == 0:
                c_star = int(np.argmax(s))
                memberships[j, c_star] = 1
                self._v_forced += 1
            else:
                self._v_positive += 1

            Xhat[:, j] = components @ memberships[j, :]

    def _greedy_update_poisson(
        self,
        X: np.ndarray,
        Xhat: np.ndarray,
        memberships: np.ndarray,  # U (m×R) if axis=0, V (n×C) if axis=1
        idx: int,  # row index i if axis=0, col index j if axis=1
        components: np.ndarray,  # G (R×n) if axis=0, H (m×C) if axis=1
        scorer_fn,  # self._poisson_delta_ll_row or _col
        beta: float,
        k_limit: int | None,
        axis: int,  # 0 = row update, 1 = col update
    ) -> None:
        """
        Greedy Poisson-membership update for one row (axis=0) or one column (axis=1).

        What it does :
        ------------
        - For a single row i (axis=0) or column j (axis=1), we re-assign its
        cluster memberships under a **Poisson log-likelihood objective**.
        - Start from the current reconstruction slice mu = Xhat[i,:] or Xhat[:,j].
        - Remove the contribution of this row/column's existing memberships so we
        can start fresh.
        - Iteratively consider adding each candidate component (row of G or column
        of H). For each candidate, compute its **log-likelihood gain** using
        scorer_fn:
            Δℓ = Δ log P(X | mu + component) - β.
        - Greedily add the component with the largest positive Δℓ, update mu, and
        repeat until no component gives positive gain or the membership budget
        k_limit is reached.
        - If nothing was chosen, force-add the single best component.
        - Finally, update the reconstruction slice in Xhat for this row/column.

        Intuition:
        ----------
        This is a sparse, greedy MAP step: assign a row/column to the set of
        clusters that most improve its Poisson log-likelihood, but penalize each
        assignment by β to prevent overfitting.
        """

        if axis == 0:
            # -------- ROW UPDATE (i = idx) --------
            i = idx
            mu = np.maximum(Xhat[i, :], 1e-9)
            if memberships[i, :].any():
                mu = np.maximum(
                    mu - components[memberships[i, :] == 1, :].sum(axis=0),
                    1e-9,
                )

            memberships[i, :] = 0
            used = np.zeros(components.shape[0], dtype=bool)
            picks = 0
            x_slice = X[i, :]

            while (k_limit is None) or (picks < k_limit):
                scores = np.array(
                    [
                        scorer_fn(x_slice, mu, components[r, :]) - beta
                        for r in range(components.shape[0])
                    ]
                )
                scores[used] = -np.inf
                r_star = int(np.argmax(scores))
                if scores[r_star] <= 0:
                    break
                memberships[i, r_star] = 1
                mu = mu + components[r_star, :]
                used[r_star] = True
                picks += 1

            if memberships[i, :].sum() == 0:
                scores = np.array(
                    [
                        scorer_fn(x_slice, mu, components[r, :]) - beta
                        for r in range(components.shape[0])
                    ]
                )
                memberships[i, int(np.argmax(scores))] = 1

            Xhat[i, :] = memberships[i, :] @ components

        else:
            # -------- COLUMN UPDATE (j = idx) --------
            j = idx
            mu = np.maximum(Xhat[:, j], 1e-9)
            if memberships[j, :].any():
                mu = np.maximum(
                    mu - components[:, memberships[j, :] == 1].sum(axis=1),
                    1e-9,
                )

            memberships[j, :] = 0
            used = np.zeros(components.shape[1], dtype=bool)
            picks = 0
            x_slice = X[:, j]

            while (k_limit is None) or (picks < k_limit):
                scores = np.array(
                    [
                        scorer_fn(x_slice, mu, components[:, c]) - beta
                        for c in range(components.shape[1])
                    ]
                )
                scores[used] = -np.inf
                c_star = int(np.argmax(scores))
                if scores[c_star] <= 0:
                    break
                memberships[j, c_star] = 1
                mu = mu + components[:, c_star]
                used[c_star] = True
                picks += 1

            if memberships[j, :].sum() == 0:
                scores = np.array(
                    [
                        scorer_fn(x_slice, mu, components[:, c]) - beta
                        for c in range(components.shape[1])
                    ]
                )
                memberships[j, int(np.argmax(scores))] = 1

            Xhat[:, j] = components @ memberships[j, :]

    def check_fitted(self):
        if (
            self.U_ is None
            or self.V_ is None
            or self.B_ is None
            or self.Xhat_ is None
        ):
            raise ValueError("Model has not been fitted yet.")

    def factors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (U, B, V) after fit()."""
        self.check_fitted()
        return self.U_, self.B_, self.V_

    def reconstruct(self) -> np.ndarray:
        """Return reconstruction Xhat = U B V^T after fit()."""
        self.check_fitted()
        return self.Xhat_

    def predict_reconstruction(self) -> np.ndarray:
        """Alias for reconstruct(); mirrors scikit-learn predict_* naming."""
        return self.reconstruct()

    def compute_tss_rss_pve(
        self, X: np.ndarray, baseline: str = "column_mean"
    ) -> Dict[str, float]:
        """
        Compute Total Sum of Squares (TSS), Residual Sum of Squares (RSS),
        and Percent Variance Explained (PVE) for the fitted model.

        Parameters
        ----------
        X : np.ndarray
            Original data matrix
        baseline : str, default="column_mean"
            Baseline for TSS computation:
            - "column_mean": TSS = ||X - column_means||²_F (per-column centering)
            - "grand_mean": TSS = ||X - grand_mean||²_F (global centering)
            - "zero": TSS = ||X||²_F (no centering)

        Returns
        -------
        dict
            Dictionary containing 'tss', 'rss', 'pve' values
        """
        self.check_fitted()
        X = np.asarray(X, dtype=np.float64)
        Xhat = self.Xhat_

        # Compute RSS (always the same)
        rss = float(np.sum((X - Xhat) ** 2))

        # Compute TSS based on baseline
        if baseline == "column_mean":
            # Per-column mean baseline (like cluster_util.py)
            Xc = X - X.mean(axis=0, keepdims=True)
            tss = float(np.sum(Xc * Xc))
        elif baseline == "grand_mean":
            # Global mean baseline (current score method)
            grand_mean = float(X.mean())
            tss = float(np.sum((X - grand_mean) ** 2))
        elif baseline == "zero":
            # No centering baseline
            tss = float(np.sum(X * X))
        else:
            raise ValueError(
                f"Unknown baseline '{baseline}'. "
                "Choose from: 'column_mean', 'grand_mean', 'zero'"
            )

        # Compute PVE
        pve = 100.0 * (1.0 - rss / max(tss, 1e-12)) if tss > 0 else np.nan

        return {"tss": tss, "rss": rss, "pve": pve, "baseline": baseline}

    def compute_tss_rss_pve_per_block(
        self,
        X: np.ndarray,
        baseline: str = "column_mean",
        return_dataframe: bool = True,
    ) -> Dict[str, Any] | pd.DataFrame:
        """
        Compute TSS, RSS, and PVE statistics per block ID.

        Parameters
        ----------
        X : np.ndarray
            Original data matrix
        baseline : str, default="column_mean"
            Baseline for TSS computation:
            - "column_mean": TSS = ||X - column_means||²_F (per-column centering)
            - "grand_mean": TSS = ||X - grand_mean||²_F (global centering)
            - "zero": TSS = ||X||²_F (no centering)
        return_dataframe : bool, default=True
            If True, return results as pandas DataFrame; if False, return as dict

        Returns
        -------
        dict or pd.DataFrame
            Per-block statistics with columns/keys:
            - block_id: Block identifier (r*C + c)
            - r, c: Row and column cluster indices
            - n_cells: Number of cells in this block
            - tss: Total sum of squares for this block
            - rss: Residual sum of squares for this block
            - pve: Percent variance explained for this block
            - baseline: Baseline method used
        """
        self.check_fitted()
        X = np.asarray(X, dtype=np.float64)
        Xhat = self.Xhat_
        B = self.B_
        I, J = X.shape
        R, C = B.shape

        # Get block assignments for each cell
        block_assignments = self.assign_cells_to_blocks(X)
        block_ids = block_assignments["block_id"]

        # Get unique block IDs (excluding -1 for unassigned)
        unique_blocks = np.unique(block_ids)
        unique_blocks = unique_blocks[unique_blocks >= 0]

        results = []

        for block_id in unique_blocks:
            # Get mask for cells in this block
            mask = block_ids == block_id
            if not mask.any():
                continue

            # Extract data for this block
            X_block = X[mask]
            Xhat_block = Xhat[mask]

            # Compute RSS for this block
            rss = float(np.sum((X_block - Xhat_block) ** 2))

            # Compute TSS for this block based on baseline
            if baseline == "column_mean":
                # Use global column means as baseline
                col_means = X.mean(axis=0)
                # Get column indices for this block
                row_indices, col_indices = np.where(mask)
                baseline_values = col_means[col_indices]
                tss = float(np.sum((X_block - baseline_values) ** 2))
            elif baseline == "grand_mean":
                grand_mean = float(X.mean())
                tss = float(np.sum((X_block - grand_mean) ** 2))
            elif baseline == "zero":
                tss = float(np.sum(X_block**2))
            else:
                raise ValueError(
                    f"Unknown baseline '{baseline}'. "
                    "Choose from: 'column_mean', 'grand_mean', 'zero'"
                )

            # Compute PVE
            pve = 100.0 * (1.0 - rss / max(tss, 1e-12)) if tss > 0 else np.nan

            # Get block coordinates
            r = int(block_id // C)
            c = int(block_id % C)

            results.append(
                {
                    "block_id": int(block_id),
                    "r": r,
                    "c": c,
                    "n_cells": int(mask.sum()),
                    "tss": tss,
                    "rss": rss,
                    "pve": pve,
                    "B_rc": float(B[r, c]),
                    "baseline": baseline,
                }
            )

        if return_dataframe:
            return (
                pd.DataFrame(results)
                .sort_values("block_id")
                .reset_index(drop=True)
            )
        else:
            return {"per_block_stats": results, "baseline": baseline}

    def score(self, X: np.ndarray) -> Dict[str, float]:
        """
        Compute diagnostics on a given X using the learned factors:
          - gaussian: MSE, RMSE, SSE, ExplainedVariance, TSS, RSS, PVE
          - poisson : Deviance (2 * Σ [ x log(x/μ) - (x-μ) ]) and NLL-like objective
          - common  : active bits, mean memberships, frac >1 (rows/cols)
        """
        self.check_fitted()
        X = np.asarray(X, dtype=np.float64)
        Xhat = self.Xhat_
        out: Dict[str, float] = {}

        active_bits = int(self.U_.sum() + self.V_.sum())
        out["active_bits"] = active_bits
        out["row_avg_memberships"] = float(self.U_.sum(axis=1).mean())
        out["col_avg_memberships"] = float(self.V_.sum(axis=1).mean())
        out["row_frac_gt1"] = float((self.U_.sum(axis=1) > 1).mean())
        out["col_frac_gt1"] = float((self.V_.sum(axis=1) > 1).mean())

        if self.loss == "gaussian":
            resid = X - Xhat
            sse = float(np.sum(resid * resid))
            mse = sse / X.size
            rmse = np.sqrt(mse)

            # Use dedicated TSS/RSS/PVE computation with grand_mean baseline for backward compatibility
            tss_rss_pve = self.compute_tss_rss_pve(X, baseline="grand_mean")
            explained_variance = (
                tss_rss_pve["pve"] / 100.0
            )  # Convert back to fraction

            out.update(
                dict(
                    sse=sse,
                    mse=mse,
                    rmse=rmse,
                    explained_variance=explained_variance,
                    tss=tss_rss_pve["tss"],
                    rss=tss_rss_pve["rss"],
                    pve=tss_rss_pve["pve"],
                )
            )
        else:
            MU = np.maximum(Xhat, 1e-12)
            # NLL-like objective (same form used in fit)
            nll_like = float((MU - X * np.log(MU)).sum())
            # Poisson deviance (saturated vs. model), defined with x log(x/μ) and x=0 handled as limit
            with np.errstate(divide="ignore", invalid="ignore"):
                term = np.where(
                    X > 0, X * (np.log(X) - np.log(MU)) - (X - MU), -(X - MU)
                )
                # classic deviance is 2 * Σ [ x log(x/μ) - (x-μ) ], equal to 2 * Σ term
            deviance = float(2.0 * term.sum())
            out.update(
                dict(poisson_nll_like=nll_like, poisson_deviance=deviance)
            )

        return out

    def get_row_clusters(self, *, as_bool: bool = False) -> np.ndarray:
        """
        Return the binary multi-hot row memberships U.

        Returns
        -------
        np.ndarray of shape (n_rows, n_row_clusters)
            Entry (i, r) is 1/True iff row i belongs to row-cluster r.
        """
        self.check_fitted()
        U = self.U_
        return U.astype(bool).copy() if as_bool else U.astype(np.int8).copy()

    def get_column_clusters(self, *, as_bool: bool = False) -> np.ndarray:
        """
        Return the binary multi-hot column memberships V.

        Returns
        -------
        np.ndarray of shape (n_cols, n_col_clusters)
            Entry (j, c) is 1/True iff column j belongs to col-cluster c.
        """
        self.check_fitted()
        V = self.V_
        return V.astype(bool).copy() if as_bool else V.astype(np.int8).copy()

    def get_params(self, deep: bool = True) -> dict:
        """scikit-learn-compatible parameter dict."""
        _ = deep  # unused; required by sklearn signature
        return {
            "n_row_clusters": self.n_row_clusters,
            "n_col_clusters": self.n_col_clusters,
            "k_row": self.k_row,
            "k_col": self.k_col,
            "loss": self.loss,
            "alpha": self.alpha,
            "beta": self.beta,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return string representation of the estimator."""
        params = self.get_params()
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        s = f"{self.__class__.__name__}({param_str})"
        if N_CHAR_MAX is not None and N_CHAR_MAX > 0 and len(s) > N_CHAR_MAX:
            return s[: N_CHAR_MAX - 3] + "..."
        return s

    def show_memberships(
        self,
        *,
        max_rows: int = 10,
        max_cols: int = 10,
        row_names=None,
        col_names=None,
        only_multi: bool = True,
    ) -> dict:
        """Show cluster membership statistics and examples (rows & columns)."""
        import numpy as np

        self.check_fitted()
        U, V = self.U_, self.V_
        I, J = U.shape[0], V.shape[0]

        if row_names is None:
            row_names = [f"row_{i}" for i in range(I)]
        if col_names is None:
            col_names = [f"col_{j}" for j in range(J)]

        # membership counts per entity
        row_counts = U.sum(axis=1)  # (# clusters per row)
        col_counts = V.sum(axis=1)  # (# clusters per column)

        # quick hists (how many entities have k active clusters)
        row_hist = np.bincount(
            row_counts.astype(int), minlength=max(1, int(row_counts.max()) + 1)
        ).tolist()
        col_hist = np.bincount(
            col_counts.astype(int), minlength=max(1, int(col_counts.max()) + 1)
        ).tolist()

        stats = {
            "row_stats": {
                "total_rows": int(row_counts.size),
                "multi_membership_rows": int((row_counts > 1).sum()),
                "rows_frac_gt1": float((row_counts > 1).mean()),
                "mean_clusters_per_row": float(row_counts.mean()),
                "max_clusters_per_row": int(row_counts.max()),
                "rows_with_no_clusters": int((row_counts == 0).sum()),
                "hist_counts_by_k": row_hist,  # index k -> count of rows with k clusters
            },
            "col_stats": {
                "total_cols": int(col_counts.size),
                "multi_membership_cols": int((col_counts > 1).sum()),
                "cols_frac_gt1": float((col_counts > 1).mean()),
                "mean_clusters_per_col": float(col_counts.mean()),
                "max_clusters_per_col": int(col_counts.max()),
                "cols_with_no_clusters": int((col_counts == 0).sum()),
                "hist_counts_by_k": col_hist,  # index k -> count of cols with k clusters
            },
        }

        # choose which entities to show
        row_mask = row_counts > 1 if only_multi else np.ones(I, dtype=bool)
        col_mask = col_counts > 1 if only_multi else np.ones(J, dtype=bool)

        multi_rows = np.flatnonzero(row_mask)[:max_rows]
        multi_cols = np.flatnonzero(col_mask)[:max_cols]

        # graceful fallback if none meet the filter
        if multi_rows.size == 0 and I > 0:
            multi_rows = np.arange(min(I, max_rows))
        if multi_cols.size == 0 and J > 0:
            multi_cols = np.arange(min(J, max_cols))

        row_examples = []
        for i in multi_rows:
            clusters = np.flatnonzero(U[i]).tolist()
            row_examples.append(
                {
                    "row_index": int(i),
                    "row_name": str(row_names[i]),
                    "num_clusters": len(clusters),
                    "cluster_ids": clusters,
                }
            )

        col_examples = []
        for j in multi_cols:
            clusters = np.flatnonzero(V[j]).tolist()
            col_examples.append(
                {
                    "col_index": int(j),
                    "col_name": str(col_names[j]),
                    "num_clusters": len(clusters),
                    "cluster_ids": clusters,
                }
            )

        stats["row_examples"] = row_examples
        stats["col_examples"] = col_examples
        return stats

    def assign_unique_blocks(
        self,
        X: np.ndarray | None = None,
        *,
        method: (
            str | None
        ) = None,  # "gaussian_delta" | "poisson_delta" | "dominant" | None(auto)
        row_names=None,
        col_names=None,
        return_frame: bool = False,
        eps: float = 1e-9,
        allowed_mask: np.ndarray | None = None,
        min_abs_B: float | None = None,
        on_empty: str = "fallback",  # "skip" | "fallback" | "raise"
    ):
        """
        Assign a unique (r*, c*) to each (i,j) consistent with X̂ = U B Vᵀ.
        Fast vectorized path for method == "dominant"; others use the scalar loop.
        """
        # --- Preconditions ---
        if (
            self.U_ is None
            or self.V_ is None
            or self.B_ is None
            or self.Xhat_ is None
        ):
            raise ValueError("Model has not been fitted yet.")
        U, V, B, Xhat = self.U_, self.V_, self.B_, self.Xhat_
        I, J = U.shape[0], V.shape[0]
        R, C = B.shape

        # --- Allowed mask via magnitude threshold ---
        if allowed_mask is None and min_abs_B is not None:
            allowed_mask = np.abs(B) >= min_abs_B
        if allowed_mask is not None and allowed_mask.shape != B.shape:
            raise ValueError(
                f"allowed_mask shape {allowed_mask.shape} must match B {B.shape}"
            )

        # --- Auto-select method ---
        if method is None:
            method = (
                "gaussian_delta"
                if (self.loss == "gaussian" and X is not None)
                else (
                    "poisson_delta"
                    if (self.loss == "poisson" and X is not None)
                    else "dominant"
                )
            )
        if method in ("gaussian_delta", "poisson_delta") and X is None:
            raise ValueError(f"Method '{method}' requires X.")

        # --- Default names for optional frame ---
        if row_names is None:
            row_names = [f"row_{i}" for i in range(I)]
        if col_names is None:
            col_names = [f"col_{j}" for j in range(J)]

        # --- Outputs ---
        r_star = np.full((I, J), -1, dtype=int)
        c_star = np.full((I, J), -1, dtype=int)

        # --- Quick exit if no memberships ---
        if (
            U.nnz == 0 or V.nnz == 0
            if hasattr(U, "nnz") and hasattr(V, "nnz")
            else (U.sum() == 0 or V.sum() == 0)
        ):
            # No active memberships anywhere
            block_id = -1 * np.ones((I, J), dtype=int)
            out = {"r_star": r_star, "c_star": c_star, "block_id": block_id}
            if return_frame:
                ii, jj = np.meshgrid(np.arange(I), np.arange(J), indexing="ij")
                out["as_frame"] = pd.DataFrame(
                    {
                        "row_i": ii.ravel(),
                        "row_name": np.array(row_names, dtype=object)[
                            ii.ravel()
                        ],
                        "col_j": jj.ravel(),
                        "col_name": np.array(col_names, dtype=object)[
                            jj.ravel()
                        ],
                        "r_star": r_star.ravel(),
                        "c_star": c_star.ravel(),
                        "block_id": (-1 * np.ones_like(r_star)).ravel(),
                    }
                )
            return out

        # =========================
        # Fast path: "dominant"
        # =========================
        if method == "dominant":
            # 1) base score matrix S (R×C)
            if self.loss == "gaussian":
                S = np.abs(B).astype(np.float64, copy=False)
            else:  # poisson
                S = B.astype(np.float64, copy=False)

            # Apply allowed_mask: disallowed -> -inf
            if allowed_mask is not None:
                S = np.where(allowed_mask, S, -np.inf)

            # Global fallback (only meaningful when we have allowed_mask)
            if allowed_mask is not None and np.isfinite(S).any():
                gr, gc = np.unravel_index(np.nanargmax(S), S.shape)
            else:
                # If no mask, any (r,c) with max S is OK as a fallback.
                if np.isfinite(S).any():
                    gr, gc = np.unravel_index(np.nanargmax(S), S.shape)
                else:
                    gr = gc = 0  # degenerate case: everything -inf

            # 2) Precompute, for every column j:
            #    M[:, j] = max_c in Cj S[:, c]
            #    Carg[:, j] = argmax_c in Cj (index in [0..C-1]) achieving that max
            V_bool = (V > 0).astype(bool)
            M = np.full((R, J), -np.inf, dtype=np.float64)
            Carg = np.full((R, J), -1, dtype=int)

            # Loop over j (J is usually much larger than I; this keeps memory light)
            for j in range(J):
                cj = np.flatnonzero(V_bool[j])
                if cj.size == 0:
                    continue
                S_sub = S[:, cj]  # (R × |Cj|)
                # argmax across c for each r
                arg_c = np.argmax(S_sub, axis=1)  # (R,)
                M[:, j] = S_sub[np.arange(R), arg_c]
                Carg[:, j] = cj[arg_c]

            # 3) For each row i, restrict to active r and take argmax over r for all j
            U_bool = (U > 0).astype(bool)
            cols = np.arange(J)

            for i in range(I):
                ri = np.flatnonzero(U_bool[i])
                if ri.size == 0:
                    continue

                # Mask M to only active r; set others to -inf
                Mi = M.copy()
                inactive = np.ones(R, dtype=bool)
                inactive[ri] = False
                Mi[inactive, :] = -np.inf

                # Best r for each j
                rbest = np.argmax(Mi, axis=0)  # (J,)
                vmax = Mi[rbest, cols]  # values at the chosen r

                # If a column j has no eligible (r,c) (all -inf), handle on_empty
                bad = ~np.isfinite(vmax)
                if np.any(bad):
                    if on_empty == "skip":
                        # leave (-1,-1)
                        pass
                    elif on_empty == "fallback":
                        r_star[i, bad] = gr
                        c_star[i, bad] = gc
                    else:
                        # raise with context on first bad j
                        j0 = int(np.flatnonzero(bad)[0])
                        raise RuntimeError(
                            f"No allowed (r,c) for cell (i={i}, j={j0})."
                        )

                # Valid columns: set winners
                good = ~bad
                if np.any(good):
                    r_star[i, good] = rbest[good]
                    c_star[i, good] = Carg[rbest[good], cols[good]]

            # Flattened block id; keep -1 for unassigned
            block_id = np.where(
                (r_star >= 0) & (c_star >= 0), r_star * C + c_star, -1
            )

            out = {"r_star": r_star, "c_star": c_star, "block_id": block_id}
            if return_frame:
                ii, jj = np.meshgrid(np.arange(I), np.arange(J), indexing="ij")
                out["as_frame"] = pd.DataFrame(
                    {
                        "row_i": ii.ravel(),
                        "row_name": np.array(row_names, dtype=object)[
                            ii.ravel()
                        ],
                        "col_j": jj.ravel(),
                        "col_name": np.array(col_names, dtype=object)[
                            jj.ravel()
                        ],
                        "r_star": r_star.ravel(),
                        "c_star": c_star.ravel(),
                        "block_id": block_id.ravel(),
                    }
                )
            return out

        # =========================
        # Scalar loop for other methods
        # =========================
        # (keeps the fixed Gaussian-delta sign & the -1 block_id semantics)
        row_active = [np.flatnonzero(U[i]).astype(int) for i in range(I)]
        col_active = [np.flatnonzero(V[j]).astype(int) for j in range(J)]

        if allowed_mask is not None:
            # global fallback uses dominant score definition
            Sglob = np.where(
                allowed_mask,
                (np.abs(B) if self.loss == "gaussian" else B),
                -np.inf,
            )
            if np.isfinite(Sglob).any():
                gr, gc = np.unravel_index(np.argmax(Sglob), Sglob.shape)
            else:
                gr = gc = 0

        for i in range(I):
            Ri = row_active[i]
            if Ri.size == 0:
                continue
            for j in range(J):
                Cj = col_active[j]
                if Cj.size == 0:
                    continue

                B_sub = B[np.ix_(Ri, Cj)]

                if method == "gaussian_delta":
                    r_ij = float(X[i, j] - Xhat[i, j])
                    scores = r_ij * B_sub - 0.5 * (B_sub * B_sub)
                else:  # "poisson_delta"
                    x_ij = float(X[i, j])
                    mu_ij = float(max(Xhat[i, j], eps))
                    denom = np.maximum(mu_ij - B_sub, eps)
                    scores = x_ij * (np.log(mu_ij) - np.log(denom)) - B_sub

                if allowed_mask is not None:
                    mask_sub = allowed_mask[np.ix_(Ri, Cj)]
                    scores = np.where(mask_sub, scores, -np.inf)

                if not np.isfinite(scores).any():
                    if on_empty == "skip":
                        continue
                    elif on_empty == "fallback":
                        r_star[i, j], c_star[i, j] = int(gr), int(gc)
                        continue
                    else:
                        raise RuntimeError(
                            f"No allowed (r,c) for cell (i={i}, j={j})."
                        )

                flat_idx = int(np.argmax(scores))
                rr, cc = divmod(flat_idx, scores.shape[1])
                r_star[i, j] = int(Ri[rr])
                c_star[i, j] = int(Cj[cc])

        block_id = np.where(
            (r_star >= 0) & (c_star >= 0), r_star * C + c_star, -1
        )
        out = {"r_star": r_star, "c_star": c_star, "block_id": block_id}

        if return_frame:
            ii, jj = np.meshgrid(np.arange(I), np.arange(J), indexing="ij")
            out["as_frame"] = pd.DataFrame(
                {
                    "row_i": ii.ravel(),
                    "row_name": np.array(row_names, dtype=object)[ii.ravel()],
                    "col_j": jj.ravel(),
                    "col_name": np.array(col_names, dtype=object)[jj.ravel()],
                    "r_star": r_star.ravel(),
                    "c_star": c_star.ravel(),
                    "block_id": block_id.ravel(),
                }
            )

        return out

    # -------- ClusterMixin compatibility --------
    def fit_predict(self, X, y=None):
        """
        Fit the model and return cluster labels for rows.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each sample. For multi-membership,
            returns the first (lowest index) cluster assignment.
        """
        self.fit(X, y)
        return self._get_row_labels()

    def _get_row_labels(self) -> np.ndarray:
        """
        Convert binary row memberships to single cluster labels.
        For multi-membership, returns the first (lowest-index) active cluster.
        If a row has no active bits, returns -1 for that row.
        """
        if self.U_ is None:
            raise ValueError("Model has not been fitted yet")

        I, _ = self.U_.shape
        labels = np.full(I, -1, dtype=int)
        # argmax on boolean chooses the FIRST True; if none True → 0, so guard with any()
        for i in range(I):
            if self.U_[i].any():
                labels[i] = int(np.argmax(self.U_[i]))
        return labels

    @property
    def labels_(self) -> np.ndarray:
        """
        Cluster labels for each sample (row).
        For multi-membership, returns the first active cluster assignment.
        """
        return self._get_row_labels()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict a single hard row-cluster label for each new row in X.

        Strategy:
        - Precompute G = B @ V^T (size R × J).
        - For each row x in X, choose r that maximizes the per-row score:
            * Gaussian:  -0.5 * ||x - G[r]||^2
            * Poisson:   sum_j [ x_j * log(mu_rj) - mu_rj ],  mu_rj = max(G[r,j], 1e-9)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted row-cluster index in [0..R-1], or -1 if factors are missing.
        """
        if self.U_ is None or self.B_ is None or self.V_ is None:
            raise ValueError("Model has not been fitted yet")

        X = check_array(
            X, dtype=np.float64, ensure_2d=True, force_all_finite="allow-nan"
        )
        X = np.nan_to_num(X)  # be robust to NaNs

        R, C = self.B_.shape
        J = self.V_.shape[0]
        if X.shape[1] != J:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was fit with {J} columns."
            )

        # Precompute cluster prototypes for rows: shape (R, J)
        G = self.B_ @ self.V_.T

        labels = np.empty(X.shape[0], dtype=int)

        if self.loss == "gaussian":
            # For each x: pick r minimizing squared distance => maximizing the negative distance
            # Vectorized: compute ||x - G_r||^2 for all r
            # We'll loop over rows to keep memory small but use vector ops inside.
            for i, x in enumerate(X):
                # dists^2 = ||G||^2 - 2 x·G^T + ||x||^2  (the last term is constant per i)
                x_dot_G = G @ x  # shape (R,)
                G_norm2 = np.einsum("rj,rj->r", G, G)
                # score = -0.5 * ||x - G_r||^2  == x·G_r - 0.5*||G_r||^2  (dropping const -0.5||x||^2)
                scores = x_dot_G - 0.5 * G_norm2
                labels[i] = int(np.argmax(scores))

        elif self.loss == "poisson":
            # Ensure positivity for Poisson mean
            MU = np.maximum(G, 1e-9)  # shape (R, J)
            LOG_MU = np.log(MU)
            for i, x in enumerate(X):
                # loglik_r = sum_j [ x_j * log(mu_rj) - mu_rj ]
                # (x may have zeros/negatives if preprocessed; clip at 0 for Poisson)
                x_clip = np.maximum(x, 0.0)
                scores = x_clip @ LOG_MU.T - MU.sum(axis=1)
                labels[i] = int(np.argmax(scores))
        else:
            raise ValueError("Unknown loss; expected 'gaussian' or 'poisson'.")

        return labels

    def predict_blocks(
        self,
        X: np.ndarray,
        *,
        method: str | None = None,
        allowed_mask: np.ndarray | None = None,
        min_abs_B: float | None = None,
        on_empty: str = "fallback",
    ):
        """
        Return per-cell assignments for a matrix X:
        - block_id: (I,J) int array with r*C + c (or -1 when skipped)
        - r_star:   (I,J) row-cluster index chosen per cell (or -1)
        - c_star:   (I,J) col-cluster index chosen per cell (or -1)
        Thin wrapper around `assign_unique_blocks` that checks shapes.
        """
        if self.V_ is None or self.B_ is None:
            raise ValueError("Model has not been fitted yet")

        X = check_array(
            X, dtype=np.float64, ensure_2d=True, force_all_finite="allow-nan"
        )
        X = np.nan_to_num(X)

        J = self.V_.shape[0]
        if X.shape[1] != J:
            raise ValueError(
                f"X has {X.shape[1]} features, but model expects {J}."
            )

        # default method per loss
        if method is None:
            method = (
                "gaussian_delta"
                if self.loss == "gaussian"
                else "poisson_delta"
            )

        return self.assign_unique_blocks(
            X,
            method=method,
            allowed_mask=allowed_mask,
            min_abs_B=min_abs_B,
            on_empty=on_empty,
        )

    def allowed_mask_from_gap(
        self, *, min_keep: int = 4, eps: float = 1e-12
    ) -> np.ndarray:
        """
        Return a boolean (R×C) mask over B_ that keeps the 'strong' blocks.

        Heuristic:
        1) Rank the |B| entries from largest to smallest (ignoring near-zeros).
        2) Find the biggest *gap* between consecutive magnitudes.
        3) Keep everything at or above the magnitude where that biggest gap occurs,
            but always keep at least `min_keep` entries.

        Parameters
        ----------
        min_keep : int
            Minimum number of block coefficients to keep, even if the gap suggests fewer.
        eps : float
            Values with |B| <= eps are treated as effectively zero and ignored for gap finding.

        Returns
        -------
        np.ndarray (dtype=bool, same shape as B_)
            True where the block is kept (allowed), False otherwise.
        """
        # Must be fitted to have B_
        if self.B_ is None:
            raise ValueError("Model has not been fitted yet")

        logger.info(
            f"Computing allowed mask from gap with min_keep={min_keep} eps={eps}"
        )
        # 1) Collect absolute values of all block weights into a flat vector
        absB = np.abs(self.B_).ravel()

        # 2) Sort descending and drop (near-)zeros so they don't affect the gap
        vals = np.sort(absB)[::-1]  # descending
        vals = vals[vals > eps]  # ignore near-zero entries

        # If nothing left after thresholding, return an all-False mask
        if vals.size == 0:
            return np.zeros_like(self.B_, dtype=bool)

        # If the number of non-negligible entries is small, keep them all
        if vals.size <= min_keep:
            logger.info(
                f"Only {vals.size} non-negligible blocks; keeping them all."
            )
            cut = vals[-1]  # the smallest of what's left
            return np.abs(self.B_) >= cut

        # 3) Compute adjacent drops (gaps) in the sorted magnitude curve:
        #    drops[i] = vals[i] - vals[i+1]
        drops = vals[:-1] - vals[1:]

        # 4) Find the largest gap and keep everything up to that index (inclusive)
        #    +1 because k items correspond to indices 0..k-1 in `vals`
        k = int(np.argmax(drops) + 1)

        # 5) Enforce minimum keep
        k = max(min_keep, k)

        # 6) Threshold: keep anything with |B| >= the k-th value
        cut = vals[k - 1]
        logger.info(f"Keeping {k} blocks with |B| >= {cut}.")
        return np.abs(self.B_) >= cut

    def keep_topk_blocks(self, k: int) -> np.ndarray:
        if self.B_ is None:
            raise ValueError("Model has not been fitted yet")

        # 1) Collect absolute values of all block weights into a flat vector
        absB = np.abs(self.B_).ravel()

        thr = np.partition(absB, -k)[-k]
        logger.info(f"Keeping top {k} blocks with |B| >= {thr}.")
        return np.abs(self.B_) >= thr

    def blockmask_to_cellmask(self, allowed_mask: np.ndarray) -> np.ndarray:
        """
        allowed_mask: (R, C) boolean or {0,1}
        Returns: (I, J) boolean cell-level mask:
        True if cell (i,j) can see at least one allowed block (r,c)
        """
        # U,V are binary
        U, V = self.U_, self.V_
        A = U.astype(int) @ allowed_mask.astype(int)  # (I, C)
        M = A @ V.astype(int).T  # (I, J)
        return M > 0

    # ----------------------------
    # Fast per-block ablation ΔLoss
    # ----------------------------

    def compute_all_block_delta_losses(
        self, X: np.ndarray, mask: np.ndarray = None
    ):

        # If B is larger, trim a view
        U, B, V = self.factors()
        # Use aligned dims from U and V
        R = U.shape[1]
        C = V.shape[1]
        # B = est.B_
        if B.shape != (R, C):
            B = B[:R, :C]

        dmat = np.zeros((R, C), dtype=float)
        for r in range(R):
            for c in range(C):
                dmat[r, c] = self.ablate_block_delta_loss(
                    X, r, c, B=B, mask=mask
                )
        return dmat

    def ablate_block_delta_loss(
        self,
        X: np.ndarray,
        r: int,
        c: int,
        B: np.ndarray,
        mask: np.ndarray = None,
    ):
        """
        ΔLoss_rc = Loss(X, Xhat_without_rc) - Loss(X, Xhat_full)
        where removing block (r,c) is a rank-1 update:
        Xhat_without = Xhat - B_rc * (U[:,r] ⊗ V[:,c])
        Positive Δ means the block is helpful.
        """
        U, _, V = self.factors()
        Xhat = self.reconstruct()
        # U, V, B, Xhat = est.U_, est.V_, Bview, est.Xhat_

        b_rc = float(B[r, c])
        if b_rc == 0.0:
            return 0.0
        loss_name = self.loss
        L_full = model_loss(X, Xhat, loss_name, mask)
        # rank-1 contribution with current U/V
        outer = np.outer(U[:, r].astype(float), V[:, c].astype(float))
        Xhat_wo = Xhat - b_rc * outer
        L_wo = model_loss(X, Xhat_wo, loss_name, mask)
        return L_wo - L_full

    def collect_block_stats(self, X, mask=None):
        U, B, V = self.factors()
        R, C = U.shape[1], V.shape[1]

        # counts
        n_rows = U.sum(axis=0).astype(int)  # (R,)
        n_cols = V.sum(axis=0).astype(int)  # (C,)

        rows = []
        for r in range(R):
            for c in range(C):
                nr, nc = int(n_rows[r]), int(n_cols[c])
                n_cells = nr * nc
                b = float(B[r, c])

                # ΔLoss (positive means helpful)
                dloss = self.ablate_block_delta_loss(X, r, c, B=B, mask=mask)

                rows.append(
                    {
                        "r": r,
                        "c": c,
                        "B_rc": b,
                        "n_rows": nr,
                        "n_cols": nc,
                        "n_cells": n_cells,
                        "delta_loss": dloss,
                        "effect_score": abs(b) * (n_cells**0.5),
                    }
                )
        return pd.DataFrame(rows)

    def filter_blocks(
        self,
        X: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        min_keep: int = 4,
        method: str = "gaussian_delta",
        return_frame: bool = False,
        keep_strategy: str | None = None,  # NEW: None = legacy behavior
        size_q: float = 0.50,
        delta_q: float = 0.50,
        min_rows: int = 1,
        min_cols: int = 1,
    ) -> Dict[str, Any]:
        """
        Filter blocks by a boolean mask or by data-driven stats.
        keep_strategy:
        - Gap: use legacy mask = self.allowed_mask_from_gap(min_keep=min_keep)
        - TopK: keep top-k blocks by absolute value
        - 'size_gap'        : elbow on n_cells
        - 'effect_gap'      : elbow on effect_score = |B| * sqrt(n_cells)
        - 'delta_then_size' : ΔLoss quantile AND size quantile (recommended)
        - 'quantiles'       : size-only quantile
        """
        if self.B_ is None:
            raise ValueError("Model has not been fitted yet")

        logger.info(f"min_keep:{min_keep}-keep_strategy: {keep_strategy}")
        if keep_strategy == "Gap":
            # legacy: whatever your current gap logic does
            logger.info("Computing allowed mask from gap.")
            allowed = (
                self.allowed_mask_from_gap(min_keep=min_keep)
                if mask is None
                else mask
            )
        elif keep_strategy == "TopK":
            logger.info("Computing allowed mask from top-k.")
            allowed = self.keep_topk_blocks(k=min_keep)
        elif keep_strategy in (
            "size_gap",
            "effect_gap",
            "delta_then_size",
            "quantiles",
        ):
            # data-driven
            logger.info("Computing allowed mask from stats.")
            stats = self.collect_block_stats(X, mask=None)
            logger.info(f"stats:\n{stats}")
            allowed = allowed_mask_from_stats(
                stats,
                rule=keep_strategy,
                size_quantile=size_q,
                delta_quantile=delta_q,
                min_rows=min_rows,
                min_cols=min_cols,
            )
            logger.info(f"allowed:\n{allowed}")
        else:
            raise ValueError(f"Unknown keep_strategy: {keep_strategy}")

        assign = self.assign_unique_blocks(
            X,
            method=method,
            allowed_mask=allowed,
            on_empty="fallback",
            return_frame=return_frame,
        )
        return assign["as_frame"] if return_frame else assign

    def get_row_col_orders(
        self,
        assign: Dict[str, Any],
        norm_data: pd.DataFrame,
    ) -> Tuple[List[int], List[int]]:
        r_star, c_star = assign["r_star"], assign["c_star"]
        R, C = self.B_.shape
        store_r = np.apply_along_axis(_mode_ignore_minus1, 1, r_star, R)
        item_c = np.apply_along_axis(_mode_ignore_minus1, 0, c_star, C)
        row_order = (
            pd.Series(store_r, index=norm_data.index)
            .sort_values()
            .index.tolist()
        )
        col_order = (
            pd.Series(item_c, index=norm_data.columns)
            .sort_values()
            .index.tolist()
        )
        return row_order, col_order

    def get_store_item_assignments(
        self,
        assign: Dict[str, Any],
        norm_data: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        r_star, c_star = assign["r_star"], assign["c_star"]
        R, C = self.B_.shape
        store_r = np.apply_along_axis(_mode_ignore_minus1, 1, r_star, R)
        item_c = np.apply_along_axis(_mode_ignore_minus1, 0, c_star, C)
        store_assign = pd.DataFrame(
            {"store_r": store_r}, index=norm_data.index
        )
        item_assign = pd.DataFrame({"item_c": item_c}, index=norm_data.columns)
        return store_assign, item_assign

    def explain_blocks(
        self,
        X: np.ndarray,
        assign: dict | None,  # optional; only to restrict which blocks to list
        row_names: np.ndarray,
        col_names: np.ndarray,
        top_k: int = 5,
        b_thresh: float = 0.0,  # ignore blocks with |B_rc| <= b_thresh when assign is None
    ) -> pd.DataFrame:
        """
        Summarize blocks using overlapping U/V memberships.
        Reports both overlap-counted and 'exclusive' (overlap-adjusted) statistics.

        exclusive weighting: each cell contributes 1/K where
        K = (# row clusters containing i) * (# col clusters containing j).
        """
        U, B, V = self.U_, self.B_, self.V_
        R, C = B.shape

        row_names = np.asarray(row_names)
        col_names = np.asarray(col_names)

        # figure out which blocks to report
        if assign is not None and "block_id" in assign:
            used = np.unique(np.asarray(assign["block_id"]))
            used = used[used >= 0]
            rc_pairs = [divmod(int(b), C) for b in used]
        else:
            # all (r, c) that are "live": some members on both sides and |B_rc| > b_thresh
            rc_pairs = []
            U_any = (U > 0).any(axis=0)  # shape (R,)
            V_any = (V > 0).any(axis=0)  # shape (C,)
            for r in range(R):
                if not U_any[r]:
                    continue
                for c in range(C):
                    if not V_any[c]:
                        continue
                    if abs(B[r, c]) > b_thresh:
                        rc_pairs.append((r, c))

        # precompute multiplicities to avoid building a full overlap matrix
        row_mult = (
            (U > 0).sum(axis=1).astype(np.int32)
        )  # (# row clusters per row i)
        col_mult = (
            (V > 0).sum(axis=1).astype(np.int32)
        )  # (# col clusters per col j)

        total_cells = X.size
        rows_out = []

        for r, c in rc_pairs:
            u_mask = U[:, r].astype(bool)  # (I,)
            v_mask = V[:, c].astype(bool)  # (J,)

            if not u_mask.any() or not v_mask.any():
                continue

            # submatrix for this block
            X_block = X[np.ix_(u_mask, v_mask)]
            n_cells = X_block.size
            if n_cells == 0:
                continue

            # overlap-counted stats
            mean_overlap = float(X_block.mean())
            # median_overlap = float(np.median(X_block))

            # exclusive stats: weight each (i,j) by 1 / (row_mult[i] * col_mult[j])
            rm = row_mult[u_mask].astype(float)[:, None]  # (Ir, 1)
            cm = col_mult[v_mask].astype(float)[None, :]  # (1, Jc)
            denom = rm * cm  # (Ir, Jc)
            # guard against zeros (shouldn't happen unless U/V empty)
            denom[denom == 0] = 1.0
            W = 1.0 / denom
            sum_w = float(W.sum())
            mean_exclusive = (
                float((W * X_block).sum() / sum_w)
                if sum_w > 0
                else mean_overlap
            )
            # no canonical "exclusive median"; we keep overlap-based median

            # coverage
            # coverage_overlap = 100.0 * (n_cells / total_cells)
            # exclusive coverage (each covered cell counts 1/K)
            exclusive_coverage = float(W.sum()) / total_cells * 100.0

            # top-k names by average within the block (more meaningful than arbitrary slices)
            row_means = X_block.mean(axis=1)
            col_means = X_block.mean(axis=0)
            top_row_idx = np.argsort(-row_means)[:top_k]
            top_col_idx = np.argsort(-col_means)[:top_k]
            stores_in_r = row_names[u_mask][top_row_idx].tolist()
            items_in_c = col_names[v_mask][top_col_idx].tolist()

            rows_out.append(
                {
                    "block_id": int(r * C + c),
                    "r": int(r),
                    "c": int(c),
                    "B_rc": float(B[r, c]),
                    "n_cells": int(n_cells),
                    # "coverage_%_overlap": coverage_overlap,
                    "weighted_coverage_%": exclusive_coverage,
                    # "mean_overlap": mean_overlap,
                    # "median_overlap": median_overlap,
                    "weighted_mean": mean_exclusive,
                    "n_stores_in_r": int(u_mask.sum()),
                    "n_items_in_c": int(v_mask.sum()),
                    "stores_in_r_topk": stores_in_r,
                    "items_in_c_topk": items_in_c,
                }
            )

        df = pd.DataFrame(rows_out)
        if not df.empty:
            # Sort by magnitude of the block weight or by exclusive mean—your choice
            df = df.reindex(
                df["B_rc"].abs().sort_values(ascending=False).index
            )
            # alternative: df = df.sort_values("mean_exclusive", ascending=False)
        return df

    @classmethod
    def factory(
        cls, **frozen_kwargs
    ) -> Callable[..., "BinaryTriFactorizationEstimator"]:
        """
        Returns a callable builder:
            builder(n_row_clusters, n_col_clusters, *, random_state=None, **overrides) -> estimator
        `frozen_kwargs` are fixed defaults; `overrides` allow per-call tweaks.
        """
        return BTFEstimatorBuilder(cls, frozen_kwargs)


class BTFEstimatorBuilder:
    """
    Picklable builder class for BinaryTriFactorizationEstimator.

    This replaces the local function approach to make it compatible with multiprocessing.
    """

    def __init__(self, estimator_class, frozen_kwargs):
        self.estimator_class = estimator_class
        self.frozen_kwargs = dict(frozen_kwargs)

    def __call__(
        self,
        n_row_clusters: int,
        n_col_clusters: int,
        *,
        random_state: Optional[int] = None,
        **overrides: Any,
    ) -> "BinaryTriFactorizationEstimator":
        """Build an estimator instance with the given parameters."""
        kw: Dict[str, Any] = dict(self.frozen_kwargs)
        kw.update(overrides)
        kw["n_row_clusters"] = int(n_row_clusters)
        kw["n_col_clusters"] = int(n_col_clusters)
        logger.info(
            f"Building estimator with n_row={n_row_clusters}, n_col={n_col_clusters}"
        )
        if random_state is not None:
            kw["random_state"] = int(random_state)
        return self.estimator_class(**kw)
