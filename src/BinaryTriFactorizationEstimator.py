# Implementation: Binary Multi-Hard Tri-Factorization (scikit-learn style)
# with Gaussian/Poisson losses
import numpy as np
from typing import Optional, Tuple, Dict, List, Any
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array
from sklearn.cluster import KMeans

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _mode_ignore_minus1(vec: np.ndarray, K: int) -> int:
    v = vec[vec >= 0]
    if v.size == 0:
        return -1
    counts = np.bincount(v, minlength=K)
    return int(np.argmax(counts))  # ties → lowest index


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
        n_row_clusters,
        n_col_clusters,
        k_row=2,
        k_col=2,
        loss="gaussian",
        alpha=1e-3,
        beta=0.0,
        max_iter=30,
        tol=1e-4,
        random_state=0,
        verbose=False,
        block_l1=0.0,
        b_inner=15,
    ):
        """
        Parameters
        ----------
        n_row_clusters : int
            Number of row clusters
        n_col_clusters : int
            Number of column clusters
        k_row : int or None, default=2
            Max active clusters per row. None for data-driven stopping.
        k_col : int or None, default=2
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
        verbose : bool, default=False
            Whether to print progress information
        block_l1 : float, default=0.0
            L1 regularization on B matrix (0 = off)
        b_inner : int, default=15
            Inner prox steps for B when block_l1 > 0
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
        self.verbose = verbose
        self.block_l1 = block_l1
        self.b_inner = b_inner

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

    # -------- B updates --------
    def _update_B_gaussian(self, X, U, V) -> np.ndarray:
        """
        Ridge-regularized least-squares for B:
          B = (U^T U + αI)^(-1) U^T X V (V^T V + αI)^(-1)
        Implemented via two solves for numerical stability (no explicit inverse).
        """
        R, C = self.n_row_clusters, self.n_col_clusters
        if self.block_l1 <= 0.0:
            UtU = U.T @ U + self.alpha * np.eye(R)
            VtV = V.T @ V + self.alpha * np.eye(C)
            M = U.T @ X @ V
            Left = np.linalg.solve(UtU, M)
            B = np.linalg.solve(VtV.T, Left.T).T
            return B

        # Prox-ISTA on: 0.5||X-UBV^T||^2 + (alpha/2)||B||^2 + block_l1*||B||_1
        if self.B_ is None:
            B = np.zeros((R, C), dtype=np.float64)
        else:
            B = self.B_.copy().astype(np.float64)

        UtU = U.T @ U
        VtV = V.T @ V
        # Lipschitz bound for gradient of the smooth part
        L = float(np.linalg.norm(UtU, 2) * np.linalg.norm(VtV, 2) + self.alpha)
        eta = 1.0 / max(L, 1e-9)
        for _ in range(max(1, self.b_inner)):
            # grad of 0.5||X-UBV^T||^2 + (alpha/2)||B||^2
            G = U.T @ (U @ B @ V.T - X) @ V + self.alpha * B
            B = B - eta * G
            # soft-threshold (prox for L1)
            thr = eta * self.block_l1
            B = np.sign(B) * np.maximum(np.abs(B) - thr, 0.0)
        return B

    def _update_B_poisson(self, X, U, V, B, n_inner: int = 15) -> np.ndarray:
        """
        Multiplicative updates for Poisson/KL with fixed U,V:
          B ← B ⊙ [ U^T (X / (U B V^T)) V ] / [ U^T 1 V + α ]
        Keeps B strictly positive to avoid numerical issues.
        """
        B = np.maximum(B, 1e-6)
        ones = np.ones_like(X, dtype=X.dtype)
        for _ in range(n_inner):
            MU = (U @ B) @ V.T
            MU = np.maximum(MU, 1e-9)
            numer = U.T @ (X / MU) @ V
            # add block_l1 to the denominator = L1 shrink on B
            denom = (U.T @ ones @ V) + self.alpha + self.block_l1
            B *= numer / np.maximum(denom, 1e-12)
            B = np.maximum(B, 1e-12)
        return B

    # -------- Toggle scoring (marginal gain) --------
    def _toggle_scores_gaussian_row(self, residual_row, G, G_norm2, beta):
        """
        Scores to add a row-bit r: S = r·g_r - 0.5||g_r||^2 - β
        - residual_row: current unexplained part of X[i, :]
        - G[r, :] is the row vector added when U[i, r] toggles 0→1
        """
        return residual_row @ G.T - 0.5 * G_norm2 - beta

    def _toggle_scores_gaussian_col(self, residual_col, H, H_norm2, beta):
        """Column analogue of the Gaussian toggle score."""
        return H.T @ residual_col - 0.5 * H_norm2 - beta

    def _poisson_delta_ll_row(self, x_row, mu_row, g_row):
        """
        Exact Δ log-likelihood for Poisson when toggling a row-bit:
          sum_j [ x log(mu+g) - (mu+g) - (x log mu - mu) ]
        """
        mu_new = np.maximum(mu_row + g_row, 1e-9)
        mu_row = np.maximum(mu_row, 1e-9)
        return (x_row * (np.log(mu_new) - np.log(mu_row)) - (mu_new - mu_row)).sum()

    def _poisson_delta_ll_col(self, x_col, mu_col, h_col):
        """Column analogue of the Poisson Δ log-likelihood."""
        mu_new = np.maximum(mu_col + h_col, 1e-9)
        mu_col = np.maximum(mu_col, 1e-9)
        return (x_col * (np.log(mu_new) - np.log(mu_col)) - (mu_new - mu_col)).sum()

    # -------- Fit --------
    def fit(self, X, y=None):
        """
        Fit the binary tri-factorization model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Validate input
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        _ = y  # Mark as intentionally unused
        I, J = X.shape
        R, C = self.n_row_clusters, self.n_col_clusters
        rng = self._rng()

        # Initialize memberships
        U = self._init_binary((I, R), self.k_row, rng)
        V = self._init_binary((J, C), self.k_col, rng)

        # Initialize B
        if self.loss == "gaussian":
            B = self._update_B_gaussian(X, U, V)
        elif self.loss == "poisson":
            B = np.abs(rng.normal(0.5, 0.1, size=(R, C))) + 0.1  # positive start
            B = self._update_B_poisson(X, U, V, B, n_inner=20)
        else:
            raise ValueError("loss must be 'gaussian' or 'poisson'")

        if self.history_flag:
            self.loss_history_ = []
        prev = np.inf

        for it in range(self.max_iter):
            # --- B-step ---
            if self.loss == "gaussian":
                B = self._update_B_gaussian(X, U, V)
            else:
                B = self._update_B_poisson(X, U, V, B, n_inner=self.b_inner)

            # --- Precompute shared quantities ---
            G = B @ V.T  # (R, J): effect of toggling a row-bit r on an entire row
            H = U @ B  # (I, C): effect of toggling a col-bit c on an entire column
            Xhat = U @ G  # (I, J) == H @ V^T

            # Norms for Gaussian toggle scoring
            if self.loss == "gaussian":
                G_norm2 = np.einsum("rj,rj->r", G, G)  # shape (R,)
                H_norm2 = np.einsum("ic,ic->c", H, H)  # shape (C,)

            # --- U-step: per-row greedy construction ---
            for i in range(I):
                if self.loss == "gaussian":
                    # Clean residual for row i (remove current recon; add back own contributions)
                    base = X[i, :] - Xhat[i, :]
                    if U[i, :].any():
                        base = base + G[U[i, :] == 1, :].sum(axis=0)

                    U[i, :] = 0
                    residual = base.copy()
                    used = np.zeros(R, dtype=bool)
                    picks = 0
                    while (self.k_row is None) or (picks < self.k_row):
                        scores = self._toggle_scores_gaussian_row(
                            residual, G, G_norm2, self.beta
                        )
                        scores[used] = -np.inf
                        r_star = int(np.argmax(scores))
                        if (
                            scores[r_star] <= 0
                        ):  # data-driven stopping when no positive gain
                            break
                        U[i, r_star] = 1
                        residual -= G[r_star, :]
                        used[r_star] = True
                        picks += 1

                    # Ensure ≥1 membership
                    if U[i, :].sum() == 0:
                        r_star = int(
                            np.argmax(
                                self._toggle_scores_gaussian_row(
                                    residual, G, G_norm2, self.beta
                                )
                            )
                        )
                        U[i, r_star] = 1

                    Xhat[i, :] = U[i, :] @ G

                else:  # Poisson
                    mu = np.maximum(Xhat[i, :], 1e-9)
                    if U[i, :].any():
                        mu = np.maximum(mu - G[U[i, :] == 1, :].sum(axis=0), 1e-9)
                    U[i, :] = 0
                    used = np.zeros(R, dtype=bool)
                    picks = 0
                    x_i = X[i, :]
                    while (self.k_row is None) or (picks < self.k_row):
                        scores = np.array(
                            [
                                self._poisson_delta_ll_row(x_i, mu, G[r, :]) - self.beta
                                for r in range(R)
                            ]
                        )
                        scores[used] = -np.inf
                        r_star = int(np.argmax(scores))
                        if scores[r_star] <= 0:
                            break
                        U[i, r_star] = 1
                        mu = mu + G[r_star, :]
                        used[r_star] = True
                        picks += 1

                    if U[i, :].sum() == 0:
                        scores = np.array(
                            [
                                self._poisson_delta_ll_row(x_i, mu, G[r, :]) - self.beta
                                for r in range(R)
                            ]
                        )
                        U[i, int(np.argmax(scores))] = 1

                    Xhat[i, :] = U[i, :] @ G

            # --- Refresh H and Xhat after U-step ---
            H = U @ B
            Xhat = H @ V.T

            # --- V-step: per-column greedy construction ---
            if self.loss == "gaussian":
                H_norm2 = np.einsum("ic,ic->c", H, H)

            for j in range(J):
                if self.loss == "gaussian":
                    base = X[:, j] - Xhat[:, j]
                    if V[j, :].any():
                        base = base + H[:, V[j, :] == 1].sum(axis=1)

                    V[j, :] = 0
                    residual = base.copy()
                    used = np.zeros(C, dtype=bool)
                    picks = 0
                    while (self.k_col is None) or (picks < self.k_col):
                        scores = self._toggle_scores_gaussian_col(
                            residual, H, H_norm2, self.beta
                        )
                        scores[used] = -np.inf
                        c_star = int(np.argmax(scores))
                        if scores[c_star] <= 0:
                            break
                        V[j, c_star] = 1
                        residual -= H[:, c_star]
                        used[c_star] = True
                        picks += 1

                    if V[j, :].sum() == 0:
                        V[
                            j,
                            int(
                                np.argmax(
                                    self._toggle_scores_gaussian_col(
                                        residual, H, H_norm2, self.beta
                                    )
                                )
                            ),
                        ] = 1

                    Xhat[:, j] = H @ V[j, :]

                else:  # Poisson
                    mu = np.maximum(Xhat[:, j], 1e-9)
                    if V[j, :].any():
                        mu = np.maximum(mu - H[:, V[j, :] == 1].sum(axis=1), 1e-9)
                    V[j, :] = 0
                    used = np.zeros(C, dtype=bool)
                    picks = 0
                    x_j = X[:, j]
                    while (self.k_col is None) or (picks < self.k_col):
                        scores = np.array(
                            [
                                self._poisson_delta_ll_col(x_j, mu, H[:, c]) - self.beta
                                for c in range(C)
                            ]
                        )
                        scores[used] = -np.inf
                        c_star = int(np.argmax(scores))
                        if scores[c_star] <= 0:
                            break
                        V[j, c_star] = 1
                        mu = mu + H[:, c_star]
                        used[c_star] = True
                        picks += 1

                    if V[j, :].sum() == 0:
                        scores = np.array(
                            [
                                self._poisson_delta_ll_col(x_j, mu, H[:, c]) - self.beta
                                for c in range(C)
                            ]
                        )
                        V[j, int(np.argmax(scores))] = 1

                    Xhat[:, j] = H @ V[j, :]

            # --- Finalize this outer iteration: recompute recon & objective ---
            G = B @ V.T
            Xhat = U @ G

            if self.loss == "gaussian":
                resid = X - Xhat
                penalty = self.beta * (U.sum() + V.sum())
                loss = 0.5 * np.sum(resid * resid) + penalty
            else:
                MU = np.maximum(Xhat, 1e-9)
                loss = float((MU - X * np.log(MU)).sum()) + self.beta * (
                    U.sum() + V.sum()
                )

            if self.history_flag:
                self.loss_history_.append(loss)
            if self.verbose:
                logger.info(f"[iter {it:02d}] loss={loss:.6e}")

            # Early stopping on relative improvement
            if np.isfinite(prev) and prev - loss < self.tol * max(1.0, prev):
                break
            prev = loss

        # Save learned factors (cast U,V to 0/1 int8)
        self.U_, self.V_, self.B_, self.Xhat_ = (
            U.astype(np.int8),
            V.astype(np.int8),
            B,
            Xhat,
        )
        return self

    # -------- Helpers / API --------
    def check_fitted(self):
        if self.U_ is None or self.V_ is None or self.B_ is None or self.Xhat_ is None:
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

    def score(self, X: np.ndarray) -> Dict[str, float]:
        """
        Compute diagnostics on a given X using the learned factors:
          - gaussian: MSE, RMSE, SSE, ExplainedVariance
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
            # Explained variance vs. mean-only model
            grand_mean = float(X.mean())
            tss = float(np.sum((X - grand_mean) ** 2))
            explained_variance = 1.0 - (sse / max(tss, 1e-12))
            out.update(
                dict(sse=sse, mse=mse, rmse=rmse, explained_variance=explained_variance)
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
            out.update(dict(poisson_nll_like=nll_like, poisson_deviance=deviance))

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
            "verbose": self.verbose,
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
        allowed_mask: (
            np.ndarray | None
        ) = None,  # NEW: boolean R×C mask of blocks you allow
        min_abs_B: (
            float | None
        ) = None,  # NEW: convenience threshold; builds allowed_mask if given
        on_empty: str = "fallback",  # "skip" | "fallback" | "raise"
    ):
        if self.U_ is None or self.V_ is None or self.B_ is None or self.Xhat_ is None:
            raise ValueError("Model has not been fitted yet.")
        U, V, B, Xhat = self.U_, self.V_, self.B_, self.Xhat_
        I, J = U.shape[0], V.shape[0]
        R, C = B.shape

        # build mask from min_abs_B if requested
        if allowed_mask is None and min_abs_B is not None:
            allowed_mask = np.abs(B) >= min_abs_B

        # auto-pick method
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

        if row_names is None:
            row_names = [f"row_{i}" for i in range(I)]
        if col_names is None:
            col_names = [f"col_{j}" for j in range(J)]

        row_active = [np.flatnonzero(U[i]).astype(int) for i in range(I)]
        col_active = [np.flatnonzero(V[j]).astype(int) for j in range(J)]

        r_star = np.full((I, J), -1, dtype=int)
        c_star = np.full((I, J), -1, dtype=int)

        # global best allowed (for fallback)
        if allowed_mask is not None:
            global_scores = np.where(
                allowed_mask,
                (np.abs(B) if self.loss == "gaussian" else B),
                -np.inf,
            )
            if np.isfinite(global_scores).any():
                gr, gc = np.unravel_index(np.argmax(global_scores), global_scores.shape)
            else:
                gr = gc = 0  # no allowed blocks at all

        for i in range(I):
            Ri = row_active[i]
            if Ri.size == 0:
                continue
            for j in range(J):
                Cj = col_active[j]
                if Cj.size == 0:
                    continue

                B_sub = B[np.ix_(Ri, Cj)]
                if method == "dominant":
                    scores = np.abs(B_sub) if self.loss == "gaussian" else B_sub
                elif method == "gaussian_delta":
                    r_ij = float(X[i, j] - Xhat[i, j])
                    scores = r_ij * B_sub + 0.5 * (B_sub * B_sub)
                else:  # "poisson_delta"
                    x_ij = float(X[i, j])
                    mu_ij = float(max(Xhat[i, j], eps))
                    denom = np.maximum(mu_ij - B_sub, eps)
                    scores = x_ij * (np.log(mu_ij) - np.log(denom)) - B_sub

                # apply allowed mask if provided
                if allowed_mask is not None:
                    mask_sub = allowed_mask[np.ix_(Ri, Cj)]
                    scores = np.where(mask_sub, scores, -np.inf)

                if not np.isfinite(scores).any():
                    if on_empty == "skip":
                        continue  # leaves -1,-1
                    elif on_empty == "fallback":
                        r_star[i, j], c_star[i, j] = int(gr), int(gc)
                        continue
                    else:
                        raise RuntimeError("No allowed (r,c) for this cell.")

                flat_idx = int(np.argmax(scores))
                rr, cc = divmod(flat_idx, scores.shape[1])
                r_star[i, j] = int(Ri[rr])
                c_star[i, j] = int(Cj[cc])

        block_id = r_star * C + c_star
        out = {"r_star": r_star, "c_star": c_star, "block_id": block_id}
        if return_frame:
            ii, jj = np.meshgrid(np.arange(I), np.arange(J), indexing="ij")
            as_frame = pd.DataFrame(
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
            out["as_frame"] = as_frame
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

        I, R = self.U_.shape
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
        X = np.nan_to_num(X, copy=False)  # be robust to NaNs

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
        X = np.nan_to_num(X, copy=False)

        J = self.V_.shape[0]
        if X.shape[1] != J:
            raise ValueError(f"X has {X.shape[1]} features, but model expects {J}.")

        # default method per loss
        if method is None:
            method = "gaussian_delta" if self.loss == "gaussian" else "poisson_delta"

        return self.assign_unique_blocks(
            X,
            method=method,
            allowed_mask=allowed_mask,
            min_abs_B=min_abs_B,
            on_empty=on_empty,
        )

    def primary_row_labels(
        self,
        X: np.ndarray,
        *,
        rule: str = "mode",  # "mode" | "first" | "argmax"
        method: str | None = None,  # passed to predict_blocks when rule="mode"
        allowed_mask: np.ndarray | None = None,
        min_abs_B: float | None = None,
        on_empty: str = "fallback",
    ) -> np.ndarray:
        """
        One label per row (store), using different rules:

        - rule="mode":    Take the most frequent r among the per-cell r_star for that row.
                            (Uses assign_unique_blocks under the hood.)
        - rule="first":   First active bit from U_ (your current labels_ behavior).
        - rule="argmax":  Argmax cluster by prototype score (same logic as predict).

        Returns
        -------
        labels : (I,) int array of row-cluster indices; -1 if no decision possible.
        """
        if rule == "first":
            return self._get_row_labels()

        if rule == "argmax":
            return self.predict(X)

        if rule != "mode":
            raise ValueError("rule must be 'mode', 'first', or 'argmax'")

        # mode of r_star across columns
        out = self.predict_blocks(
            X,
            method=method,
            allowed_mask=allowed_mask,
            min_abs_B=min_abs_B,
            on_empty=on_empty,
        )
        r_star = out["r_star"]  # (I,J)
        R = self.B_.shape[0]

        labels = np.apply_along_axis(_mode_ignore_minus1, 1, r_star, R)
        return labels

    def primary_column_labels(
        self,
        X: np.ndarray,
        *,
        rule: str = "mode",  # "mode" | "first" | "argmax"
        method: str | None = None,
        allowed_mask: np.ndarray | None = None,
        min_abs_B: float | None = None,
        on_empty: str = "fallback",
    ) -> np.ndarray:
        """
        One label per column (item).

        - rule="mode":    Take most frequent c among per-cell c_star for that column.
        - rule="first":   First active bit from V_.
        - rule="argmax":  Choose c that best explains the column vs. U,B (mirror of predict).

        Returns
        -------
        labels : (J,) int array of column-cluster indices; -1 if no decision possible.
        """
        if rule == "first":
            if self.V_ is None:
                raise ValueError("Model has not been fitted yet")
            J, C = self.V_.shape
            labs = np.full(J, -1, dtype=int)
            for j in range(J):
                if self.V_[j].any():
                    labs[j] = int(np.argmax(self.V_[j]))
            return labs

        if rule == "argmax":
            # Prototype for columns is H = U @ B  (I×C). For a column x (I,),
            # pick c maximizing Gaussian/Poisson score against H[:,c].
            if self.U_ is None or self.B_ is None:
                raise ValueError("Model has not been fitted yet")
            H = self.U_ @ self.B_  # (I,C)
            X = check_array(
                X, dtype=np.float64, ensure_2d=True, force_all_finite="allow-nan"
            )
            X = np.nan_to_num(X, copy=False)
            I, C = H.shape
            if X.shape[0] != I:
                raise ValueError(f"X has {X.shape[0]} rows, but model expects {I}.")
            labels = np.empty(X.shape[1], dtype=int)
            if self.loss == "gaussian":
                H_norm2 = np.einsum("ic,ic->c", H, H)  # (C,)
                for j in range(X.shape[1]):
                    x = X[:, j]
                    scores = H.T @ x - 0.5 * H_norm2
                    labels[j] = int(np.argmax(scores))
            else:
                MU = np.maximum(H, 1e-9)  # (I,C)
                LOG_MU = np.log(MU)
                for j in range(X.shape[1]):
                    x = np.maximum(X[:, j], 0.0)
                    labels[j] = int(np.argmax(x @ LOG_MU - MU.sum(axis=0)))
            return labels

        # rule="mode"
        out = self.predict_blocks(
            X,
            method=method,
            allowed_mask=allowed_mask,
            min_abs_B=min_abs_B,
            on_empty=on_empty,
        )
        c_star = out["c_star"]  # (I,J)
        C = self.B_.shape[1]
        labels = np.apply_along_axis(_mode_ignore_minus1, 0, c_star, C)
        return labels

    def allowed_mask_from_gap(
        self, *, min_keep: int = 4, eps: float = 1e-12
    ) -> np.ndarray:
        """Binary mask over B selecting 'strong' blocks via the largest |B| gap."""
        if self.B_ is None:
            raise ValueError("Model has not been fitted yet")
        absB = np.abs(self.B_).ravel()
        vals = np.sort(absB)[::-1]
        vals = vals[vals > eps]
        if vals.size == 0:
            return np.zeros_like(self.B_, dtype=bool)
        if vals.size <= min_keep:
            cut = vals[-1]
            return np.abs(self.B_) >= cut
        drops = vals[:-1] - vals[1:]
        k = int(np.argmax(drops) + 1)
        k = max(min_keep, k)
        cut = vals[k - 1]
        return np.abs(self.B_) >= cut

    def filter_blocks(
        self,
        X: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        min_keep: int = 4,
        method: str = "gaussian_delta",
    ) -> Dict[str, Any]:
        """
        Filter the blocks by a boolean mask.
        """
        if self.B_ is None:
            raise ValueError("Model has not been fitted yet")
        if mask is None:
            mask = self.allowed_mask_from_gap(min_keep=min_keep)

        # per-cell assignment with the mask
        assign = self.assign_unique_blocks(
            X, method=method, allowed_mask=mask, on_empty="fallback"
        )
        return assign

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
            pd.Series(store_r, index=norm_data.index).sort_values().index.tolist()
        )
        col_order = (
            pd.Series(item_c, index=norm_data.columns).sort_values().index.tolist()
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
        store_assign = pd.DataFrame({"store_r": store_r}, index=norm_data.index)
        item_assign = pd.DataFrame({"item_c": item_c}, index=norm_data.columns)
        return store_assign, item_assign

    def explain_blocks(
        self,
        X: np.ndarray,
        assign: pd.DataFrame,
        row_names: np.ndarray,
        col_names: np.ndarray,
        top_k: int = 5,
    ) -> pd.DataFrame:
        used = np.unique(assign["block_id"])
        used = used[used >= 0]
        rows = []
        U, B, V = self.U_, self.B_, self.V_
        R, C = B.shape
        for b in used:
            r, c = int(b // C), int(b % C)
            cells = assign["block_id"] == b
            rows.append(
                {
                    "block_id": int(b),
                    "r": r,
                    "c": c,
                    "B_rc": float(B[r, c]),
                    "n_cells": int(cells.sum()),
                    "coverage_%": 100.0 * cells.mean(),
                    "mean": float(X[cells].mean()),
                    "median": float(np.median(X[cells])),
                    "stores_in_r": [
                        row_names[i] for i in np.where(U[:, r] == 1)[0][:top_k]
                    ],
                    "items_in_c": [
                        col_names[j] for j in np.where(V[:, c] == 1)[0][:top_k]
                    ],
                }
            )
        return pd.DataFrame(rows).sort_values("B_rc", ascending=False)


# --- small utilities ---------------------------------------------------------


def get_normalized_assignments(
    assign: Dict[str, Any],
    norm_data: pd.DataFrame,
    *,
    row_col: str = "store",
    col_col: str = "item",
    value_col: str = "growth_rate_1",
) -> pd.DataFrame:

    block_id_mat = assign["block_id"]

    # long (store,item,block_id)
    blk_long = (
        pd.DataFrame(block_id_mat, index=norm_data.index, columns=norm_data.columns)
        .stack()
        .rename("block_id")
        .reset_index()
        .rename(columns={"level_0": row_col, "level_1": col_col})
    )

    # long values from the SAME matrix used for clustering
    df = (
        norm_data.stack()
        .rename(value_col)
        .reset_index()
        .rename(columns={"level_0": row_col, "level_1": col_col})
    )

    # store, item, value, block_id (no date column)
    df = df.merge(blk_long, on=[row_col, col_col], how="left")

    return df


def merge_assignments(
    df: pd.DataFrame,
    assign: Dict[str, Any],
    norm_data: pd.DataFrame,
    *,
    row_col: str = "store",
    col_col: str = "item",
    value_col: str = "growth_rate_1",
) -> pd.DataFrame:

    block_id_mat = assign["block_id"]

    # long (store,item,block_id)
    blk_long = (
        pd.DataFrame(block_id_mat, index=norm_data.index, columns=norm_data.columns)
        .stack()
        .rename("block_id")
        .reset_index()
        .rename(columns={"level_0": row_col, "level_1": col_col})
    )

    # long values from the SAME matrix used for clustering
    val_long = (
        norm_data.stack()
        .rename(value_col)
        .reset_index()
        .rename(columns={"level_0": row_col, "level_1": col_col})
    )

    # plotting table: store, item, value, block_id (no date column)
    val_long = val_long.merge(blk_long, on=[row_col, col_col], how="left")

    # attach per-(store,item) block_id
    val_long = val_long[["store", "item", "block_id"]].drop_duplicates()

    df = df.merge(val_long, on=["store", "item"], how="left", validate="m:1")
    return df


def summarize_blocks_for_merge(
    est: BinaryTriFactorizationEstimator,
    X: np.ndarray,
    assign: Dict[str, Any],
) -> pd.DataFrame:
    """Per-block stats from the current assignment."""
    R, C = est.B_.shape
    bid = assign["block_id"]
    used = np.unique(bid)
    used = used[used >= 0]
    rows = []
    for b in used:
        mask = bid == b
        vals = X[mask]
        r, c = int(b // C), int(b % C)
        rows.append(
            {
                "block_id": int(b),
                "r": r,
                "c": c,
                "B_rc": float(est.B_[r, c]),
                "n_cells": int(mask.sum()),
                "coverage_%": 100.0 * mask.mean(),
                "mean": float(vals.mean()) if vals.size else np.nan,
                "median": float(np.median(vals)) if vals.size else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("block_id").reset_index(drop=True)


def explain_coverage(
    est: BinaryTriFactorizationEstimator,
    assign: Dict[str, Any],
) -> pd.DataFrame:

    # per-block coverage as a matrix
    cov = pd.DataFrame(
        [
            [
                (assign["block_id"] == (r * est.B_.shape[1] + c)).sum()
                for c in range(est.B_.shape[1])
            ]
            for r in range(est.B_.shape[0])
        ],
        index=[f"r{r}" for r in range(est.B_.shape[0])],
        columns=[f"c{c}" for c in range(est.B_.shape[1])],
    )
    cov = cov.style.background_gradient(cmap="Blues", axis=None)
    return cov


def merge_blocks_by_stat(
    assign: Dict[str, Any],
    df: pd.DataFrame,
    norm_data: pd.DataFrame,
    est: BinaryTriFactorizationEstimator,
    *,
    stat: str = "mean",
    scheme: str = "sign",
    k: int = 2,
) -> pd.DataFrame:
    """
    Merge active blocks using a single statistic of their cells.

    stat:   "mean" or "median"   (drives grouping)
    scheme: "sign" | "kmeans"
      - "sign": positives vs negatives (by chosen stat)
      - "kmeans": k-means in 1D on the chosen stat (k groups)
    """
    summ = summarize_blocks_for_merge(est, norm_data.to_numpy(), assign)
    stat_col = "mean" if stat == "mean" else "median"
    s = summ[stat_col].to_numpy()

    if scheme == "sign":
        # 0 = nonnegative, 1 = negative → two groups
        merged = (s < 0).astype(int)
        # order: group 0 (>=0) first, group 1 (<0) next
        if stat == "mean":
            order = np.argsort(
                [
                    s[merged == g].mean() if (merged == g).any() else np.inf
                    for g in [0, 1]
                ]
            )
        else:
            medians = [
                np.median(s[merged == g]) if (merged == g).any() else np.inf
                for g in [0, 1]
            ]
            order = np.argsort(np.array(medians))
        remap = {int(g_old): int(g_new) for g_new, g_old in enumerate(order)}
        summ["merged_id"] = [remap[int(g)] for g in merged]

    elif scheme == "kmeans":
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labs = km.fit_predict(s.reshape(-1, 1))
        # reorder clusters by mean stat so ids are consistent
        if stat == "mean":
            stat_df = pd.Series(s).groupby(labs).mean().sort_values()
        else:
            stat_df = pd.Series(s).groupby(labs).median().sort_values()
        remap = {int(old): int(new) for new, old in enumerate(stat_df.index)}
        summ["merged_id"] = [remap[int(g)] for g in labs]

    else:
        raise ValueError("scheme must be 'sign' or 'kmeans'")

    # map original block_id -> merged_id
    mapping = dict(zip(summ["block_id"].tolist(), summ["merged_id"].tolist()))

    # build merged block-id matrix
    bid = assign["block_id"].copy()
    bid_merged = np.full_like(bid, fill_value=-1, dtype=int)
    # vectorized map
    flat = bid.ravel()
    out = np.array([mapping.get(int(b), -1) for b in flat], dtype=int)
    bid_merged = out.reshape(bid.shape)

    # Use bid_merged
    df_assign = (
        pd.DataFrame(bid_merged, index=norm_data.index, columns=norm_data.columns)
        .stack()
        .rename("merged_block_id")
        .reset_index()
        .rename(columns={"level_0": "store", "level_1": "item"})
    )

    # attach to your training df
    df = df.drop(columns=["block_id"], errors="ignore").merge(
        df_assign[["store", "item", "merged_block_id"]],
        on=["store", "item"],
        how="left",
        validate="m:1",
    )
    df.rename(columns={"merged_block_id": "block_id"}, inplace=True)
    return df


def _mode_min(s: pd.Series) -> int:
    m = s.mode()
    return int(m.min()) if not m.empty else -1


def get_row_col_orders(df: pd.DataFrame) -> Tuple[List[int], List[int]]:
    row_order = (
        df.groupby("store")["block_id"].agg(_mode_min).sort_values().index.tolist()
    )
    col_order = (
        df.groupby("item")["block_id"].agg(_mode_min).sort_values().index.tolist()
    )
    return row_order, col_order
