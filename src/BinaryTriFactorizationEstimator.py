# Implementation: Binary Multi-Hard Tri-Factorization (scikit-learn style)
# with Gaussian/Poisson losses
import numpy as np
from typing import Optional, Tuple, Dict, List, Any, Callable
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
        r"""
        Update B under a squared-error (Gaussian) objective.

        We assume the **separable ridge** regularizer on UB and BVᵀ:
            minimize_B  (1/2)||X - U B Vᵀ||_F^2
                        + (α/2)||U B||_F^2 + (α/2)||B Vᵀ||_F^2         (★)

        First-order condition for (★):
            (UᵀU + αI) B (VᵀV + αI) = Uᵀ X V

        Closed form (computed via two linear solves, not explicit inverses):
            B = (UᵀU + αI)^{-1} Uᵀ X V (VᵀV + αI)^{-1}                (1)

        If instead you use **ridge on B itself**:
            minimize_B (1/2)||X - U B Vᵀ||_F^2 + (α/2)||B||_F^2       (†)

        then the normal equation is a Sylvester-type system:
            (UᵀU) B (VᵀV) + α B = Uᵀ X V                              (2)

        (2) should be solved via a Sylvester/Kronecker method or eigentrick,
        not with (1). The code below implements (1) (separable ridge).
        """

        R, C = self.n_row_clusters, self.n_col_clusters

        # --- Case 1: no L1, pure separable ridge -> use the two-solve closed form (1)
        if self.block_l1 <= 0.0:
            # UtU = UᵀU + αI_k,  VtV = VᵀV + αI_ℓ
            UtU = U.T @ U + self.alpha * np.eye(R)
            VtV = V.T @ V + self.alpha * np.eye(C)
            # M = Uᵀ X V
            M = U.T @ X @ V
            # Left = (UᵀU + αI)^{-1} M  (solve left system)
            Left = np.linalg.solve(UtU, M)
            # B = Left (VᵀV + αI)^{-1}  (solve right system via transposed solve)
            B = np.linalg.solve(VtV.T, Left.T).T
            return B

        # --- Case 2: add block-wise L1 on B -> solve
        #     minimize_B (1/2)||X - U B Vᵀ||_F^2
        #               + (α/2)||B||_F^2
        #               + λ ||B||_1                                      (3)
        #
        # Smooth part f(B) = (1/2)||X - U B Vᵀ||_F^2 + (α/2)||B||_F^2
        # Gradient:
        #     ∇f(B) = Uᵀ( U B Vᵀ - X )V + α B                           (4)
        # Lipschitz constant L for ∇f:
        #     L ≤ ||UᵀU||_2 · ||VᵀV||_2 + α                            (5)
        # Proximal gradient (ISTA) step with step size η = 1/L:
        #     B ← B - η ∇f(B)                                           (6)
        # Soft-thresholding (prox of λ||B||_1):
        #     B ← S_{ηλ}(B)   where S_{τ}(z) = sign(z)·max(|z|-τ,0)     (7)

        # warm start
        if self.B_ is None:
            B = np.zeros((R, C), dtype=np.float64)
        else:
            B = self.B_.copy().astype(np.float64)

        UtU = U.T @ U
        VtV = V.T @ V

        # Lipschitz bound (5)
        L = float(np.linalg.norm(UtU, 2) * np.linalg.norm(VtV, 2) + self.alpha)
        eta = 1.0 / max(L, 1e-12)

        for _ in range(max(1, self.b_inner)):
            # Gradient (4)
            G = U.T @ (U @ B @ V.T - X) @ V + self.alpha * B
            # Gradient step (6)
            B = B - eta * G
            # L1 prox (7)
            thr = eta * self.block_l1
            B = np.sign(B) * np.maximum(np.abs(B) - thr, 0.0)

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
                # G_norm2 is a vector of length R, each entry is the squared
                # ℓ2-norm of a row (template) in G.
                G_norm2 = np.einsum("rj,rj->r", G, G)  # shape (R,)
                # H_norm2 is a vector of length C, each entry is the squared
                # ℓ2-norm of a column (template) in H.
                H_norm2 = np.einsum("ic,ic->c", H, H)  # shape (C,)

            # --- U-step: per-row greedy construction ---
            for i in range(I):
                if self.loss == "gaussian":
                    # Clean residual for row i (remove current recon; add back own contributions)
                    # Row i update (your Block1):
                    self._greedy_update_gaussian(
                        X=X,
                        Xhat=Xhat,
                        memberships=U,
                        idx=i,
                        components=G,  # R×n
                        comp_norm2=G_norm2,
                        k_limit=self.k_row,
                        scorer_fn=self._toggle_scores_gaussian_row,
                        beta=self.beta,
                        axis=0,
                    )
                else:  # Poisson
                    self._greedy_update_poisson(
                        X=X,
                        Xhat=Xhat,
                        memberships=U,
                        idx=i,
                        components=G,  # R×n
                        scorer_fn=self._poisson_delta_ll_row,
                        beta=self.beta,
                        k_limit=self.k_row,
                        axis=0,
                    )

            # --- Refresh H and Xhat after U-step ---
            H = U @ B
            Xhat = H @ V.T

            # --- V-step: per-column greedy construction ---
            if self.loss == "gaussian":
                H_norm2 = np.einsum("ic,ic->c", H, H)

            for j in range(J):
                if self.loss == "gaussian":
                    self._greedy_update_gaussian(
                        X=X,
                        Xhat=Xhat,
                        memberships=V,
                        idx=j,
                        components=H,  # m×C
                        comp_norm2=H_norm2,
                        k_limit=self.k_col,
                        scorer_fn=self._toggle_scores_gaussian_col,
                        beta=self.beta,
                        axis=1,
                    )

                else:  # Poisson
                    self._greedy_update_poisson(
                        X=X,
                        Xhat=Xhat,
                        memberships=V,
                        idx=j,
                        components=H,  # m×C
                        scorer_fn=self._poisson_delta_ll_col,
                        beta=self.beta,
                        k_limit=self.k_col,
                        axis=1,
                    )

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
    def _greedy_update_gaussian(
        self,
        X: np.ndarray,
        Xhat: np.ndarray,
        memberships: np.ndarray,  # U (m×R) if axis=0, or V (n×C) if axis=1
        idx: int,  # row index (axis=0) or column index (axis=1)
        components: np.ndarray,  # G (R×n) if axis=0, or H (m×C) if axis=1
        comp_norm2: np.ndarray,  # G_norm2 or H_norm2 (squared ℓ2 norms per component)
        k_limit: int | None,  # k_row or k_col (max active memberships)
        scorer_fn,  # _toggle_scores_gaussian_row or _toggle_scores_gaussian_col
        beta: float,
        axis: int,  # 0 = update a row i; 1 = update a column j
    ) -> None:
        r"""
        Goal: For one row (or one column) at a time, choose which clusters it belongs to (binary, possibly multiple)
        so that the slice of the data is well explained by a small sum of fixed templates, while penalizing unnecessary picks by β.

        How:
        1. It computes a clean residual for that slice (remove current recon, add back its own effects).
        2. It greedily picks components with the largest marginal gain in reducing squared error (minus a penalty),
        only accepting those that yield positive gain.
        3. It updates the residual after each pick.
        4. It ensures at least one membership if nothing had positive gain.
        5. It recomputes the slice of the reconstruction from the chosen memberships.

        Why greedy? It’s a fast approximate solver for a sparse,
        combinatorial selection problem arising from the Gaussian objective;
        with the typical gain formula in (3), each step guarantees
        a non-increasing reconstruction error (until no positive gain remains).

        Purpose
        -------
        Greedy (one-slice) update of **binary, possibly-overlapping memberships**
        for either a single **row i** (axis=0) or a single **column j** (axis=1) in a
        Gaussian tri-factorization model. It chooses a small set of components
        (clusters) that best explain that slice of X, under a squared-error objective
        with an optional per-component penalty β.

        Model & Notation
        ----------------
        Data matrix: X ∈ ℝ^{m×n} with reconstruction X̂ = U B Vᵀ.
        During the greedy step we hold "component templates" fixed:

        • Row update (axis=0): components = G ∈ ℝ^{R×n},  where G[r,:] is the template
            contributed to row i when U[i,r]=1.  Reconstruction slice: X̂[i,:] = U[i,:] G.

        • Column update (axis=1): components = H ∈ ℝ^{m×C}, where H[:,c] is the template
            contributed to column j when V[j,c]=1. Reconstruction slice: X̂[:,j] = H V[j,:].

        For a slice s (row i or column j), define the **current base residual**
        by removing the present reconstruction for that slice and adding back its
        own contributions (so we can re-select memberships from a clean slate):

            Row i:   b = X[i,:] - X̂[i,:] + ∑_{r: U[i,r]=1} G[r,:]                  (1)
            Col j:   b = X[:,j] - X̂[:,j] + ∑_{c: V[j,c]=1} H[:,c]                  (2)

        We then greedily add components one-by-one, each time picking the index
        k* that maximizes a **gain score** s_k computed by `scorer_fn`. Under a
        Gaussian (least-squares) objective with a per-selection penalty β, a
        common gain for a candidate template t_k is:

            s_k = ⟨residual, t_k⟩  -  ½(‖t_k‖₂² + β),                               (3)

        where residual is updated as we add templates. (Your `scorer_fn` computes
        these scores; `comp_norm2` typically supplies ‖t_k‖₂².)

        We accept only **positive-gain** additions:

            k* = argmax_k s_k,   add if  s_{k*} > 0,                                (4)

        updating the residual:

            residual ← residual - t_{k*}.                                           (5)

        We repeat until we reach the cardinality limit (k_limit) or no positive
        gain remains. If nothing was selected, we force one membership by taking
        argmax_k s_k once.

        Finally, we refresh the reconstruction slice from the new memberships:

            Row i:  X̂[i,:] ← U[i,:] G,      Col j:  X̂[:,j] ← H V[j,:].            (6)
        """
        if axis == 0:
            # ---------- ROW UPDATE (i = idx) ----------
            i = idx
            # base residual (Eq. 1)
            base = X[i, :] - Xhat[i, :]
            if memberships[i, :].any():
                base = base + components[memberships[i, :] == 1, :].sum(axis=0)

            # clear memberships for this row, start greedy loop
            memberships[i, :] = 0
            residual = base.copy()

            R = components.shape[0]
            used = np.zeros(R, dtype=bool)
            picks = 0
            while (k_limit is None) or (picks < k_limit):
                # gain scores s_k (Eq. 3) from your scorer
                scores = scorer_fn(
                    residual, components, comp_norm2, beta
                )  # shape: (R,)
                scores[used] = -np.inf
                r_star = int(np.argmax(scores))
                # accept only positive-gain additions (Eq. 4)
                if scores[r_star] <= 0:
                    break
                memberships[i, r_star] = 1
                # update residual by subtracting chosen template (Eq. 5)
                residual -= components[r_star, :]
                used[r_star] = True
                picks += 1

            # ensure at least one membership (fallback argmax of scores)
            if memberships[i, :].sum() == 0:
                scores = scorer_fn(residual, components, comp_norm2, beta)
                memberships[i, int(np.argmax(scores))] = 1

            # refresh reconstruction slice (Eq. 6)
            Xhat[i, :] = memberships[i, :] @ components

        else:
            # ---------- COLUMN UPDATE (j = idx) ----------
            j = idx
            # base residual (Eq. 2)
            base = X[:, j] - Xhat[:, j]
            if memberships[j, :].any():
                base = base + components[:, memberships[j, :] == 1].sum(axis=1)

            memberships[j, :] = 0
            residual = base.copy()

            C = components.shape[1]
            used = np.zeros(C, dtype=bool)
            picks = 0
            while (k_limit is None) or (picks < k_limit):
                # gain scores s_k (Eq. 3) from your scorer
                scores = scorer_fn(
                    residual, components, comp_norm2, beta
                )  # shape: (C,)
                scores[used] = -np.inf
                c_star = int(np.argmax(scores))
                if scores[c_star] <= 0:
                    break
                memberships[j, c_star] = 1
                # update residual (Eq. 5)
                residual -= components[:, c_star]
                used[c_star] = True
                picks += 1

            if memberships[j, :].sum() == 0:
                scores = scorer_fn(residual, components, comp_norm2, beta)
                memberships[j, int(np.argmax(scores))] = 1

            # refresh reconstruction slice (Eq. 6)
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
                    mu - components[memberships[i, :] == 1, :].sum(axis=0), 1e-9
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
                    mu - components[:, memberships[j, :] == 1].sum(axis=1), 1e-9
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
    ) -> Dict[str, Any]:
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
            import pandas as pd

            return pd.DataFrame(results).sort_values("block_id").reset_index(drop=True)
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
            explained_variance = tss_rss_pve["pve"] / 100.0  # Convert back to fraction

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
        allowed_mask: np.ndarray | None = None,
        min_abs_B: float | None = None,
        on_empty: str = "fallback",  # "skip" | "fallback" | "raise"
    ):
        r"""
        Assign a **unique (row-block r*, col-block c*)** to every cell (i,j) of X
        consistent with the current tri-factorization X̂ = U B Vᵀ.

        Plain-English:
        • For each cell (i,j), only blocks (r,c) where row cluster r is active for i (U[i,r]=1)
            and column cluster c is active for j (V[j,c]=1) are eligible.
        • We score each eligible block using one of three modes and choose the best one:
            1) "dominant"       : pick the block with largest |B_rc| (Gaussian) or largest B_rc (Poisson).
            2) "gaussian_delta" : use a squared-error-based local score.
            3) "poisson_delta"  : use a Poisson log-likelihood-based local score.
        • Optionally restrict choices with `allowed_mask` (or via `min_abs_B`), and follow
            a policy if a cell has no eligible blocks.

        Notation used in the scoring:
        x_ij   := X[i,j]            (only needed for delta methods)
        x̂_ij  := Xhat[i,j]         (current reconstruction)
        b      := B[r,c]            (candidate block weight)

        Scoring formulas (per (i,j) and per candidate (r,c)):

        (A) Dominant (magnitude / sign-only preference):
            score_dom(rc) =
                { |b|    if self.loss == "gaussian"
                {  b     if self.loss == "poisson"

        (B) Gaussian delta (local squared-error heuristic):
            r_ij := x_ij - x̂_ij                                             (residual)
            score_gauss(rc) = r_ij * b + 0.5 * b^2
            (Heuristic ranking derived from the local effect of attributing (i,j) to block (r,c)
            under an L2 objective; larger is better.)

        (C) Poisson delta (exact scalar Δ log-likelihood for a Poisson mean change):
            μ_ij := max(x̂_ij, eps)                                         (keep μ>0)
            Consider attributing (i,j) to (r,c) by **removing** b from μ (so μ - b is the
            counterfactual mean owned by other blocks). The gain of choosing (r,c) as owner is:
                Δℓ = ℓ(x_ij | μ_ij) - ℓ(x_ij | μ_ij - b)
                    = x_ij * [log(μ_ij) - log(μ_ij - b)] - b
            score_pois(rc) = Δℓ, with μ_ij - b floored at eps to avoid log(0).

        Returns:
        dict with:
            - "r_star": (I×J) chosen row-block indices
            - "c_star": (I×J) chosen col-block indices
            - "block_id": (I×J) flat ids = r_star * C + c_star
            - "as_frame" (optional): tidy DataFrame of assignments
        """
        # --- Preconditions ---
        if self.U_ is None or self.V_ is None or self.B_ is None or self.Xhat_ is None:
            raise ValueError("Model has not been fitted yet.")
        U, V, B, Xhat = self.U_, self.V_, self.B_, self.Xhat_
        I, J = U.shape[0], V.shape[0]
        R, C = B.shape

        # --- Optional allowed mask derived from a magnitude threshold on B ---
        # allowed_mask[r,c] = True iff |B[r,c]| >= min_abs_B
        if allowed_mask is None and min_abs_B is not None:
            allowed_mask = np.abs(B) >= min_abs_B

        # --- Auto-select method if not provided ---
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

        # --- Default row/col names for output frame ---
        if row_names is None:
            row_names = [f"row_{i}" for i in range(I)]
        if col_names is None:
            col_names = [f"col_{j}" for j in range(J)]

        # --- Active memberships per row/col: R_i = {r: U[i,r]=1}, C_j = {c: V[j,c]=1} ---
        row_active = [np.flatnonzero(U[i]).astype(int) for i in range(I)]
        col_active = [np.flatnonzero(V[j]).astype(int) for j in range(J)]

        # --- Outputs to fill ---
        r_star = np.full((I, J), -1, dtype=int)
        c_star = np.full((I, J), -1, dtype=int)

        # --- Global fallback: the best allowed block overall (used when a cell has none allowed) ---
        if allowed_mask is not None:
            # global_scores mirrors (A): |B| for Gaussian, B for Poisson
            global_scores = np.where(
                allowed_mask,
                (np.abs(B) if self.loss == "gaussian" else B),
                -np.inf,
            )
            if np.isfinite(global_scores).any():
                gr, gc = np.unravel_index(np.argmax(global_scores), global_scores.shape)
            else:
                gr = gc = 0  # no allowed blocks at all (arbitrary fallback)

        # --- Main loop over cells (i,j) ---
        for i in range(I):
            Ri = row_active[i]
            if Ri.size == 0:
                # no active row clusters => cannot assign a block owner for any (i,·)
                continue
            for j in range(J):
                Cj = col_active[j]
                if Cj.size == 0:
                    # no active col clusters => cannot assign a block owner for (·,j)
                    continue

                # Candidate submatrix of B visible to this (i,j): B_sub ∈ R^{|Ri| × |Cj|}
                B_sub = B[np.ix_(Ri, Cj)]

                # --- Compute scores for each (r,c) in Ri×Cj according to the chosen method ---
                if method == "dominant":
                    # (A) Dominant: |b| for Gaussian, b for Poisson
                    scores = np.abs(B_sub) if self.loss == "gaussian" else B_sub

                elif method == "gaussian_delta":
                    # (B) Gaussian delta: score = r_ij * b + 0.5 * b^2, where r_ij = x_ij - x̂_ij
                    r_ij = float(X[i, j] - Xhat[i, j])
                    scores = r_ij * B_sub + 0.5 * (B_sub * B_sub)

                else:  # "poisson_delta"
                    # (C) Poisson delta: Δℓ = x_ij * (log μ - log(μ - b)) - b
                    x_ij = float(X[i, j])
                    mu_ij = float(max(Xhat[i, j], eps))  # μ := max(x̂_ij, eps)
                    denom = np.maximum(mu_ij - B_sub, eps)  # μ - b, floored at eps
                    scores = x_ij * (np.log(mu_ij) - np.log(denom)) - B_sub

                # --- Apply allowed_mask if provided (mask disallowed candidates with -inf) ---
                if allowed_mask is not None:
                    mask_sub = allowed_mask[np.ix_(Ri, Cj)]
                    scores = np.where(mask_sub, scores, -np.inf)

                # --- Handle the case with no finite candidates for this cell (i,j) ---
                if not np.isfinite(scores).any():
                    if on_empty == "skip":
                        # leave r_star[i,j], c_star[i,j] as -1
                        continue
                    elif on_empty == "fallback":
                        # assign the global best allowed block (gr,gc)
                        r_star[i, j], c_star[i, j] = int(gr), int(gc)
                        continue
                    else:
                        raise RuntimeError("No allowed (r,c) for this cell.")

                # --- Pick argmax over the |Ri|×|Cj| grid and map back to global (r*,c*) ---
                flat_idx = int(
                    np.argmax(scores)
                )  # flattens scores to 1D and returns the index of the maximum score.
                rr, cc = divmod(
                    flat_idx, scores.shape[1]
                )  # rr = row index, cc = column index
                # Map back to global (r*,c*) indices using the active lists Ri, Cj
                # (rr,cc) are indices into the submatrix B_sub
                r_star[i, j] = int(Ri[rr])
                c_star[i, j] = int(Cj[cc])

        # --- Flattened block id for convenience: block_id = r* * C + c* ---
        block_id = r_star * C + c_star

        out = {"r_star": r_star, "c_star": c_star, "block_id": block_id}

        # --- Optional tidy frame for downstream analysis / plotting ---
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
            X = np.nan_to_num(X)
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
        return np.abs(self.B_) >= cut

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

    def filter_blocks(
        self,
        X: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        min_keep: int = 4,
        method: str = "gaussian_delta",
        return_frame: bool = False,
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
            X,
            method=method,
            allowed_mask=mask,
            on_empty="fallback",
            return_frame=return_frame,
        )
        if return_frame:
            return assign["as_frame"]
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
        if random_state is not None:
            kw["random_state"] = int(random_state)
        return self.estimator_class(**kw)
