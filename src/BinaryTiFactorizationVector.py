import numpy as np
from typing import Optional, Tuple


class BinaryTriFactorizationVectors:
    """
    X ≈ sum_{r,c} U[:, r][:, None, None] * V[:, c][None, :, None] * B[r, c, :]
      - X: (I, J, D) matrix of D-dim vectors per (i, j)
      - U: (I, R) binary multi-memberships (rows)
      - V: (J, C) binary multi-memberships (cols)
      - B: (R, C, D) vector-valued block parameters

    Loss (gaussian):
        0.5 * ||X - \hat{X}||_F^2 + β (||U||_0 + ||V||_0)
    """

    def __init__(
        self,
        R: int,
        C: int,
        beta: float = 0.0,
        max_iters: int = 50,
        inner_iters_B: int = 3,
        inner_iters_UV: int = 1,
        random_state: Optional[int] = 0,
        init_density: float = 0.2,  # initial prob of membership 1
        verbose: bool = False,
    ):
        self.R = R
        self.C = C
        self.beta = beta
        self.max_iters = max_iters
        self.inner_iters_B = inner_iters_B
        self.inner_iters_UV = inner_iters_UV
        self.random_state = np.random.RandomState(random_state)
        self.init_density = init_density
        self.verbose = verbose

        self.U_ = None  # (I, R) {0,1}
        self.V_ = None  # (J, C) {0,1}
        self.B_ = None  # (R, C, D)

    @staticmethod
    def _predict(U: np.ndarray, B: np.ndarray, V: np.ndarray) -> np.ndarray:
        # U: (I,R), B: (R,C,D), V: (J,C)
        # -> (I,J,D) : \hat{X}_{i j d} = sum_{r,c} U_{i r} V_{j c} B_{r c d}
        # einsum keeps it clean and efficient
        return np.einsum("ir,rcd,jc->ijd", U, B, V, optimize=True)

    @staticmethod
    def _sqerr(X: np.ndarray, Xhat: np.ndarray) -> float:
        return 0.5 * np.sum((X - Xhat) ** 2)

    def _loss(self, X, U, V, B, Xhat=None) -> float:
        if Xhat is None:
            Xhat = self._predict(U, B, V)
        return self._sqerr(X, Xhat) + self.beta * (
            np.count_nonzero(U) + np.count_nonzero(V)
        )

    def _init_UV(self, I: int, J: int):
        U = (self.random_state.rand(I, self.R) < self.init_density).astype(
            np.uint8
        )
        V = (self.random_state.rand(J, self.C) < self.init_density).astype(
            np.uint8
        )
        # Avoid all-zero rows/cols:
        if (U.sum(axis=1) == 0).any():
            idx = np.where(U.sum(axis=1) == 0)[0]
            U[idx, self.random_state.randint(self.R, size=len(idx))] = 1
        if (V.sum(axis=1) == 0).any():
            idx = np.where(V.sum(axis=1) == 0)[0]
            V[idx, self.random_state.randint(self.C, size=len(idx))] = 1
        return U, V

    def _init_B(
        self, X: np.ndarray, U: np.ndarray, V: np.ndarray
    ) -> np.ndarray:
        I, J, D = X.shape
        B = np.zeros((self.R, self.C, D), dtype=X.dtype)
        # Simple non-overlap-style init: average of entries supporting (r,c)
        # with weights W_ij^{r,c} = U_ir * V_jc
        W_ir = U.astype(np.float32)  # (I,R)
        W_jc = V.astype(np.float32)  # (J,C)

        for r in range(self.R):
            for c in range(self.C):
                w_ij = np.outer(W_ir[:, r], W_jc[:, c])  # (I,J)
                s = w_ij.sum()
                if s > 0:
                    # weighted mean over (i,j)
                    num = (X * w_ij[:, :, None]).sum(axis=(0, 1))
                    B[r, c, :] = num / s
                else:
                    # random small init if block is empty
                    B[r, c, :] = 0.01 * self.random_state.randn(X.shape[2])
        return B

    def _update_B_coord_descent(
        self, X: np.ndarray, U: np.ndarray, V: np.ndarray, B: np.ndarray
    ):
        """
        Coordinate-descent update for B with overlaps handled via residuals.
        For each (r,c):
           Let W = outer(U[:,r], V[:,c])  (I,J)
           R = X - (Xhat - W * B_rc)      (I,J,D)  # residual if we 'remove' this block
           B_rc = sum_{i,j} W_ij * R_ij / sum_{i,j} W_ij
        """
        I, J, D = X.shape
        for _ in range(self.inner_iters_B):
            Xhat = self._predict(U, B, V)  # (I,J,D)
            for r in range(self.R):
                u_r = U[:, r]  # (I,)
                if not u_r.any():
                    continue
                for c in range(self.C):
                    v_c = V[:, c]  # (J,)
                    if not v_c.any():
                        continue
                    W = np.outer(u_r, v_c).astype(X.dtype)  # (I,J)
                    s = W.sum()
                    if s == 0:
                        continue
                    # Remove this block's current contribution, then re-fit it
                    # R = X - (Xhat - W * B_rc)
                    R = X - (Xhat - W[:, :, None] * B[r, c][None, None, :])
                    B[r, c, :] = (R * W[:, :, None]).sum(axis=(0, 1)) / s
        return B

    def _delta_loss_for_flip_U(self, X, U, V, B, i: int, r: int) -> float:
        # Compute loss change if we flip U[i, r] (0->1 or 1->0), holding others fixed.
        before = U[i, r]
        U[i, r] = 1 - before
        Xhat_new = self._predict(U, B, V)
        loss_new = self._sqerr(X, Xhat_new) + self.beta * (
            np.count_nonzero(U) + np.count_nonzero(V)
        )
        U[i, r] = before
        return loss_new

    def _delta_loss_for_flip_V(self, X, U, V, B, j: int, c: int) -> float:
        before = V[j, c]
        V[j, c] = 1 - before
        Xhat_new = self._predict(U, B, V)
        loss_new = self._sqerr(X, Xhat_new) + self.beta * (
            np.count_nonzero(U) + np.count_nonzero(V)
        )
        V[j, c] = before
        return loss_new

    def fit(self, X: np.ndarray):
        assert X.ndim == 3, "X must be (I, J, D)"
        I, J, D = X.shape

        # --- init history containers (compatible with old plotting code) ---
        if not hasattr(self, "loss_history_"):
            self.loss_history_ = []
        if not hasattr(self, "RecError"):
            self.RecError = []
        if not hasattr(self, "RelChange"):
            self.RelChange = []
        # optional gate
        record_history = getattr(self, "history_flag", True)

        U, V = self._init_UV(I, J)
        B = self._init_B(X, U, V)

        prev_loss = np.inf
        for it in range(self.max_iters):
            # --- strengthen B given current U,V ---
            B = self._update_B_coord_descent(X, U, V, B)

            # --- Greedy flips with B refresh after any accepted move ---
            for _ in range(self.inner_iters_UV):
                improved = False
                base_loss = self._loss(X, U, V, B)

                # rows
                for i in range(I):
                    for r in range(self.R):
                        new_loss = self._delta_loss_for_flip_U(
                            X, U, V, B, i, r
                        )
                        if new_loss + 1e-12 < base_loss:
                            U[i, r] = 1 - U[i, r]  # accept
                            base_loss = new_loss
                            # re-fit only a little to make 0->1 moves viable
                            B = self._update_B_coord_descent(X, U, V, B)
                            improved = True

                # ensure each row has at least one 1 (optional but stabilizing)
                row_empty = U.sum(axis=1) == 0
                if row_empty.any():
                    U[
                        row_empty,
                        self.random_state.randint(
                            self.R, size=row_empty.sum()
                        ),
                    ] = 1
                    B = self._update_B_coord_descent(X, U, V, B)

                # cols
                for j in range(J):
                    for c in range(self.C):
                        new_loss = self._delta_loss_for_flip_V(
                            X, U, V, B, j, c
                        )
                        if new_loss + 1e-12 < base_loss:
                            V[j, c] = 1 - V[j, c]  # accept
                            base_loss = new_loss
                            B = self._update_B_coord_descent(X, U, V, B)
                            improved = True

                # ensure each col has at least one 1 (optional)
                col_empty = V.sum(axis=1) == 0
                if col_empty.any():
                    V[
                        col_empty,
                        self.random_state.randint(
                            self.C, size=col_empty.sum()
                        ),
                    ] = 1
                    B = self._update_B_coord_descent(X, U, V, B)

                if not improved:
                    break  # no flips helped this inner round

            # --- end of outer iteration: compute current recon & losses ---
            Xhat = self._predict(U, B, V)

            rec_err = self._sqerr(X, Xhat)  # SSE only
            cur_loss = rec_err + self.beta * (
                np.count_nonzero(U) + np.count_nonzero(V)
            )

            # relative improvement vs previous outer-iter loss
            rel = (prev_loss - cur_loss) / max(abs(prev_loss), 1.0)

            # --- record history for plotting ---
            if record_history:
                self.loss_history_.append(float(cur_loss))
                self.RecError.append(float(rec_err))
                self.RelChange.append(
                    float(max(rel, 0.0))
                )  # keep non-negative for log plots

            if getattr(self, "verbose", False):
                print(
                    f"[it {it:03d}] loss={cur_loss:.4f} Δrel={rel:.3e} ||U||0={np.count_nonzero(U)} ||V||0={np.count_nonzero(V)}"
                )

            # monotonic guard / stop
            if cur_loss > prev_loss - 1e-6:
                break
            prev_loss = cur_loss

        self.U_, self.V_, self.B_ = U.astype(np.uint8), V.astype(np.uint8), B
        # (optional) store last reconstruction if you want to inspect later:
        self.Xhat_ = Xhat
        return self

    # def fit(self, X: np.ndarray):
    #     assert X.ndim == 3, "X must be (I, J, D)"
    #     I, J, D = X.shape

    #     U, V = self._init_UV(I, J)
    #     B = self._init_B(X, U, V)

    #     prev_loss = np.inf
    #     for it in range(self.max_iters):
    #         # --- strengthen B given current U,V ---
    #         B = self._update_B_coord_descent(X, U, V, B)

    #         # --- Greedy flips with B refresh after any accepted move ---
    #         for _ in range(self.inner_iters_UV):
    #             improved = False
    #             base_loss = self._loss(X, U, V, B)

    #             # rows
    #             for i in range(I):
    #                 for r in range(self.R):
    #                     new_loss = self._delta_loss_for_flip_U(
    #                         X, U, V, B, i, r
    #                     )
    #                     if new_loss + 1e-12 < base_loss:
    #                         U[i, r] = 1 - U[i, r]  # accept
    #                         base_loss = new_loss
    #                         # re-fit only a little to make 0->1 moves viable
    #                         B = self._update_B_coord_descent(X, U, V, B)
    #                         improved = True

    #             # ensure each row has at least one 1 (optional but stabilizing)
    #             row_empty = U.sum(axis=1) == 0
    #             if row_empty.any():
    #                 U[
    #                     row_empty,
    #                     self.random_state.randint(
    #                         self.R, size=row_empty.sum()
    #                     ),
    #                 ] = 1
    #                 B = self._update_B_coord_descent(X, U, V, B)

    #             # cols
    #             for j in range(J):
    #                 for c in range(self.C):
    #                     new_loss = self._delta_loss_for_flip_V(
    #                         X, U, V, B, j, c
    #                     )
    #                     if new_loss + 1e-12 < base_loss:
    #                         V[j, c] = 1 - V[j, c]  # accept
    #                         base_loss = new_loss
    #                         B = self._update_B_coord_descent(X, U, V, B)
    #                         improved = True

    #             # ensure each col has at least one 1 (optional)
    #             col_empty = V.sum(axis=1) == 0
    #             if col_empty.any():
    #                 V[
    #                     col_empty,
    #                     self.random_state.randint(
    #                         self.C, size=col_empty.sum()
    #                     ),
    #                 ] = 1
    #                 B = self._update_B_coord_descent(X, U, V, B)

    #             if not improved:
    #                 break  # no flips helped this inner round

    #         Xhat = self._predict(U, B, V)
    #         cur_loss = self._sqerr(X, Xhat) + self.beta * (
    #             np.count_nonzero(U) + np.count_nonzero(V)
    #         )

    #         if self.verbose:
    #             rel = (prev_loss - cur_loss) / max(prev_loss, 1.0)
    #             print(
    #                 f"[it {it:03d}] loss={cur_loss:.4f} Δrel={rel:.3e} ||U||0={np.count_nonzero(U)} ||V||0={np.count_nonzero(V)}"
    #             )

    #         if cur_loss > prev_loss - 1e-6:
    #             break
    #         prev_loss = cur_loss

    #     self.U_, self.V_, self.B_ = U.astype(np.uint8), V.astype(np.uint8), B
    #     return self

    def predict(self) -> np.ndarray:
        return self._predict(self.U_, self.B_, self.V_)

    def score(self, X: np.ndarray) -> float:
        """Negative loss for sklearn-style 'higher-is-better'."""
        Xhat = self.predict()
        return -self._loss(X, self.U_, self.V_, self.B_, Xhat=Xhat)
