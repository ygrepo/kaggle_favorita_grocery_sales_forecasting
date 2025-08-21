import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from sklearn.cluster._bicluster import BiclusterMixin  # scikit-learn >=1.2

# If the import path differs in your version, adjust accordingly.


# # -------- helpers --------
# def huber_loss(residuals, delta=1.0):
#     a = np.abs(residuals)
#     quad = 0.5 * (a**2)
#     lin = delta * (a - 0.5 * delta)
#     return np.where(a <= delta, quad, lin)


# def _update_C_tied(X, U, V, *, stat="mean"):
#     I, J = X.shape
#     P = U.shape[1]
#     Q = V.shape[1]
#     sums = U.T @ X @ V
#     counts = (U.T @ np.ones((I, 1))) @ (np.ones((1, J)) @ V)
#     if stat == "mean":
#         C = sums / np.maximum(counts, 1e-12)
#         C[counts == 0] = 0.0
#         return C
#     elif stat == "median":
#         C = np.zeros((P, Q), dtype=X.dtype)
#         row_assign = np.argmax(U, axis=1)
#         col_assign = np.argmax(V, axis=1)
#         for p in range(P):
#             rmask = row_assign == p
#             if not rmask.any():
#                 continue
#             Xp = X[rmask]
#             for q in range(Q):
#                 cmask = col_assign == q
#                 if not cmask.any():
#                     continue
#                 block = Xp[:, cmask]
#                 if block.size:
#                     C[p, q] = np.median(block)
#         return C
#     else:
#         raise ValueError("stat must be 'mean' or 'median'")


# def _update_U_tied(X, C, V, norm="l2", huber_delta=1.0):
#     I, J = X.shape
#     prototypes = C @ V.T  # (P, J)
#     R = X[:, None, :] - prototypes[None, :, :]  # (I,P,J)
#     if norm == "l2":
#         errs = np.sum(R * R, axis=2)
#     elif norm == "l1":
#         errs = np.sum(np.abs(R), axis=2)
#     elif norm == "huber":
#         errs = np.sum(huber_loss(R, delta=huber_delta), axis=2)
#     elif norm == "mav_ratio":
#         mae = np.sum(np.abs(R), axis=2)  # (I,P)
#         mav = np.sum(np.abs(X), axis=1, keepdims=True)  # (I,1)
#         errs = mae / np.maximum(mav, 1e-12)
#     else:
#         raise ValueError("Unsupported norm: 'l1','l2','huber','mav_ratio'")
#     U = np.zeros((I, prototypes.shape[0]), dtype=int)
#     U[np.arange(I), np.argmin(errs, axis=1)] = 1
#     return U


# def _update_V_tied(X, U, C, norm="l2", huber_delta=1.0):
#     bases = U @ C  # (I, Q)
#     R = X.T[:, None, :] - bases.T[None, :, :]  # (J,Q,I)
#     if norm == "l2":
#         errs = np.sum(R * R, axis=2)
#     elif norm == "l1":
#         errs = np.sum(np.abs(R), axis=2)
#     elif norm == "huber":
#         errs = np.sum(huber_loss(R, delta=huber_delta), axis=2)
#     elif norm == "mav_ratio":
#         mae = np.sum(np.abs(R), axis=2)  # (J,Q)
#         mav = np.sum(np.abs(X), axis=0, keepdims=True).T  # (J,1)
#         errs = mae / np.maximum(mav, 1e-12)
#     else:
#         raise ValueError("Unsupported norm: 'l1','l2','huber','mav_ratio'")
#     V = np.zeros((X.shape[1], bases.shape[1]), dtype=int)
#     V[np.arange(X.shape[1]), np.argmin(errs, axis=1)] = 1
#     return V


# def l2_objective(X, U, C, V):
#     R = X - (U @ C @ V.T)
#     return float(np.sum(R * R))


# def l2_objective_trace(X, U, C, V):
#     X2 = float(np.sum(X * X))
#     term2 = -2.0 * float(np.trace(V.T @ X.T @ U @ C))
#     UU = np.diag(np.sum(U, axis=0))
#     VV = np.diag(np.sum(V, axis=0))
#     term3 = float(np.trace(C.T @ UU @ C @ VV))
#     return X2 + term2 + term3


# def l1_objective(X, U, C, V):
#     R = X - (U @ C @ V.T)
#     return float(np.sum(np.abs(R)))


# def mav_objective(X, U, C, V, axis=None):
#     Xhat = U @ C @ V.T
#     R_abs = np.abs(X - Xhat)
#     X_abs = np.abs(X)
#     if axis is None:
#         return float(R_abs.sum() / max(X_abs.sum(), 1e-12))
#     if axis == 0:
#         num = R_abs.sum(axis=0)
#         den = np.maximum(X_abs.sum(axis=0), 1e-12)
#         return float(np.mean(num / den))
#     if axis == 1:
#         num = R_abs.sum(axis=1)
#         den = np.maximum(X_abs.sum(axis=1), 1e-12)
#         return float(np.mean(num / den))
#     raise ValueError("axis must be None, 0, or 1")


# def cluster_counts(U):
#     return np.sum(U, axis=0)


# def cluster_balance_penalty(U, V, alpha_empty=0.0, alpha_var=0.0):
#     cr = cluster_counts(U)
#     cc = cluster_counts(V)
#     empty_rows = np.sum(cr == 0)
#     empty_cols = np.sum(cc == 0)
#     penalty_empty = alpha_empty * (empty_rows + empty_cols)
#     Ir = cr.sum() if cr.sum() > 0 else 1.0
#     Ic = cc.sum() if cc.sum() > 0 else 1.0
#     var_rows = np.var(cr / Ir)
#     var_cols = np.var(cc / Ic)
#     penalty_var = alpha_var * (var_rows + var_cols)
#     return float(penalty_empty + penalty_var)


# # ---------- Online sufficient-stats shell ----------
# class _OnlineState:
#     def __init__(self, X, U, V):
#         self.X = X
#         self.I, self.J = X.shape
#         self.P = U.shape[1]
#         self.Q = V.shape[1]
#         self.U = U.copy()
#         self.V = V.copy()
#         self.block_sums = np.zeros((self.P, self.Q), dtype=float)
#         self.block_counts = np.zeros((self.P, self.Q), dtype=float)
#         self._rebuild()

#     def _rebuild(self):
#         self.block_sums.fill(0.0)
#         self.block_counts.fill(0.0)
#         ra = np.argmax(self.U, axis=1)
#         ca = np.argmax(self.V, axis=1)
#         for p in range(self.P):
#             rmask = ra == p
#             if not rmask.any():
#                 continue
#             Xp = self.X[rmask]
#             for q in range(self.Q):
#                 cmask = ca == q
#                 if not cmask.any():
#                     continue
#                 blk = Xp[:, cmask]
#                 self.block_sums[p, q] += blk.sum()
#                 self.block_counts[p, q] += blk.size

#     def current_C(self):
#         C = self.block_sums / np.maximum(self.block_counts, 1e-12)
#         C[self.block_counts == 0] = 0.0
#         return C

#     def update_rows(self, rows, U_new_rows, V, Q):
#         # Only rows whose cluster changed touch stats
#         U_old_rows = self.U[rows]
#         ra_old = np.argmax(U_old_rows, axis=1)
#         ra_new = np.argmax(U_new_rows, axis=1)
#         ca = np.argmax(V, axis=1)
#         for k, i in enumerate(rows):
#             p_old = ra_old[k]
#             p_new = ra_new[k]
#             if p_old == p_new:  # no change
#                 continue
#             xi = self.X[i]
#             for q in range(Q):
#                 cmask = ca == q
#                 if cmask.any():
#                     vals = xi[cmask]
#                     self.block_sums[p_old, q] -= vals.sum()
#                     self.block_counts[p_old, q] -= vals.size
#                     self.block_sums[p_new, q] += vals.sum()
#                     self.block_counts[p_new, q] += vals.size
#         self.U[rows] = U_new_rows

#     def update_cols(self, cols, V_new_cols, U, P):
#         V_old_cols = self.V[cols]
#         ca_old = np.argmax(V_old_cols, axis=1)
#         ca_new = np.argmax(V_new_cols, axis=1)
#         ra = np.argmax(U, axis=1)
#         for k, j in enumerate(cols):
#             q_old = ca_old[k]
#             q_new = ca_new[k]
#             if q_old == q_new:
#                 continue
#             xj = self.X[:, j]
#             for p in range(P):
#                 rmask = ra == p
#                 if rmask.any():
#                     vals = xj[rmask]
#                     self.block_sums[p, q_old] -= vals.sum()
#                     self.block_counts[p, q_old] -= vals.size
#                     self.block_sums[p, q_new] += vals.sum()
#                     self.block_counts[p, q_new] += vals.size
#         self.V[cols] = V_new_cols


# ---------- helpers (unchanged except new tie-break + warm-start utils) ----------
def huber_loss(residuals, delta=1.0):
    a = np.abs(residuals)
    quad = 0.5 * (a**2)
    lin = delta * (a - 0.5 * delta)
    return np.where(a <= delta, quad, lin)


def _one_hot_from_labels(labels, n_clusters, rng):
    """
    labels: shape (N,), int in [0, n_clusters-1] or -1 for 'random'.
    Returns one-hot (N, n_clusters).
    """
    labels = np.asarray(labels, dtype=int)
    N = labels.shape[0]
    out = np.zeros((N, n_clusters), dtype=int)

    # Randomly fill -1
    mask = labels < 0
    if mask.any():
        labels = labels.copy()
        labels[mask] = rng.integers(0, n_clusters, size=mask.sum())

    out[np.arange(N), labels] = 1
    return out


def _tiebreak_assign_min(
    errs_row, candidates, counts, rng, mode="first", balance_power=1.0
):
    """
    Resolve a tie for one sample.
    - errs_row: (K,) error values for this sample (unused except to define 'candidates')
    - candidates: 1D array of tied cluster indices
    - counts: (K,) current cluster counts (for 'balance')
    """
    if mode == "first":
        return candidates[0]
    elif mode == "random":
        return rng.choice(candidates)
    elif mode == "balance":
        # prefer lower counts**balance_power
        if counts is None:
            return candidates[0]
        weights = counts[candidates].astype(float)
        # lower is better ⇒ pick argmin(weights**power); break ties randomly
        scores = np.power(np.maximum(weights, 1e-12), balance_power)
        m = np.min(scores)
        best = candidates[np.where(np.isclose(scores, m))[0]]
        return rng.choice(best)
    else:
        raise ValueError("tie_breaker must be 'first', 'random', or 'balance'")


def _argmin_with_tiebreak(errs, counts, rng, tie_breaker="first", balance_power=1.0):
    """
    errs: (N, K) matrix of errors. Returns indices (N,) with custom tie-breaker.
    counts: (K,) counts array for 'balance' mode; can be None for others.
    """
    # baseline argmin
    mins = np.min(errs, axis=1, keepdims=True)
    ties = np.isclose(errs, mins)
    # Fast path: no ties for most rows
    if not np.any(ties.sum(axis=1) > 1):
        return np.argmin(errs, axis=1)

    # Slow path: resolve per-row ties
    N, K = errs.shape
    idx = np.empty(N, dtype=int)
    for i in range(N):
        tie_idx = np.flatnonzero(ties[i])
        if tie_idx.size == 1:
            idx[i] = tie_idx[0]
        else:
            idx[i] = _tiebreak_assign_min(
                errs[i],
                tie_idx,
                counts,
                rng,
                mode=tie_breaker,
                balance_power=balance_power,
            )
    return idx


def _update_C_tied(X, U, V, *, stat="mean"):
    I, J = X.shape
    P = U.shape[1]
    Q = V.shape[1]
    sums = U.T @ X @ V
    counts = (U.T @ np.ones((I, 1))) @ (np.ones((1, J)) @ V)
    if stat == "mean":
        C = sums / np.maximum(counts, 1e-12)
        C[counts == 0] = 0.0
        return C
    elif stat == "median":
        C = np.zeros((P, Q), dtype=X.dtype)
        row_assign = np.argmax(U, axis=1)
        col_assign = np.argmax(V, axis=1)
        for p in range(P):
            rmask = row_assign == p
            if not rmask.any():
                continue
            Xp = X[rmask]
            for q in range(Q):
                cmask = col_assign == q
                if not cmask.any():
                    continue
                block = Xp[:, cmask]
                if block.size:
                    C[p, q] = np.median(block)
        return C
    else:
        raise ValueError("stat must be 'mean' or 'median'")


def _update_U_tied(
    X,
    C,
    V,
    *,
    norm="l2",
    huber_delta=1.0,
    tie_breaker="first",
    balance_power=1.0,
    counts=None,
    rng=None,
):
    I, J = X.shape
    prototypes = C @ V.T  # (P, J)
    R = X[:, None, :] - prototypes[None, :, :]  # (I,P,J)
    if norm == "l2":
        errs = np.sum(R * R, axis=2)
    elif norm == "l1":
        errs = np.sum(np.abs(R), axis=2)
    elif norm == "huber":
        errs = np.sum(huber_loss(R, delta=huber_delta), axis=2)
    elif norm == "mav_ratio":
        mae = np.sum(np.abs(R), axis=2)
        mav = np.sum(np.abs(X), axis=1, keepdims=True)
        errs = mae / np.maximum(mav, 1e-12)
    else:
        raise ValueError("Unsupported norm: 'l1','l2','huber','mav_ratio'")
    # tie-break
    rng = np.random.default_rng() if rng is None else rng
    winners = _argmin_with_tiebreak(
        errs, counts, rng, tie_breaker=tie_breaker, balance_power=balance_power
    )
    U = np.zeros((I, prototypes.shape[0]), dtype=int)
    U[np.arange(I), winners] = 1
    return U


def _update_V_tied(
    X,
    U,
    C,
    *,
    norm="l2",
    huber_delta=1.0,
    tie_breaker="first",
    balance_power=1.0,
    counts=None,
    rng=None,
):
    bases = U @ C  # (I, Q)
    R = X.T[:, None, :] - bases.T[None, :, :]  # (J,Q,I)
    if norm == "l2":
        errs = np.sum(R * R, axis=2)
    elif norm == "l1":
        errs = np.sum(np.abs(R), axis=2)
    elif norm == "huber":
        errs = np.sum(huber_loss(R, delta=huber_delta), axis=2)
    elif norm == "mav_ratio":
        mae = np.sum(np.abs(R), axis=2)
        mav = np.sum(np.abs(X), axis=0, keepdims=True).T
        errs = mae / np.maximum(mav, 1e-12)
    else:
        raise ValueError("Unsupported norm: 'l1','l2','huber','mav_ratio'")
    rng = np.random.default_rng() if rng is None else rng
    winners = _argmin_with_tiebreak(
        errs, counts, rng, tie_breaker=tie_breaker, balance_power=balance_power
    )
    V = np.zeros((X.shape[1], bases.shape[1]), dtype=int)
    V[np.arange(X.shape[1]), winners] = 1
    return V


def l2_objective(X, U, C, V):
    R = X - (U @ C @ V.T)
    return float(np.sum(R * R))


def l2_objective_trace(X, U, C, V):
    X2 = float(np.sum(X * X))
    term2 = -2.0 * float(np.trace(V.T @ X.T @ U @ C))
    UU = np.diag(np.sum(U, axis=0))
    VV = np.diag(np.sum(V, axis=0))
    term3 = float(np.trace(C.T @ UU @ C @ VV))
    return X2 + term2 + term3


def l1_objective(X, U, C, V):
    R = X - (U @ C @ V.T)
    return float(np.sum(np.abs(R)))


def mav_objective(X, U, C, V, axis=None):
    Xhat = U @ C @ V.T
    R_abs = np.abs(X - Xhat)
    X_abs = np.abs(X)
    if axis is None:
        return float(R_abs.sum() / max(X_abs.sum(), 1e-12))
    if axis == 0:
        num = R_abs.sum(axis=0)
        den = np.maximum(X_abs.sum(axis=0), 1e-12)
        return float(np.mean(num / den))
    if axis == 1:
        num = R_abs.sum(axis=1)
        den = np.maximum(X_abs.sum(axis=1), 1e-12)
        return float(np.mean(num / den))
    raise ValueError("axis must be None, 0, or 1")


def cluster_counts(U):
    return np.sum(U, axis=0)


def cluster_balance_penalty(U, V, alpha_empty=0.0, alpha_var=0.0):
    cr = cluster_counts(U)
    cc = cluster_counts(V)
    empty_rows = np.sum(cr == 0)
    empty_cols = np.sum(cc == 0)
    penalty_empty = alpha_empty * (empty_rows + empty_cols)
    Ir = cr.sum() if cr.sum() > 0 else 1.0
    Ic = cc.sum() if cc.sum() > 0 else 1.0
    var_rows = np.var(cr / Ir)
    var_cols = np.var(cc / Ic)
    penalty_var = alpha_var * (var_rows + var_cols)
    return float(penalty_empty + penalty_var)


class _OnlineState:
    def __init__(self, X, U, V):
        self.X = X
        self.I, self.J = X.shape
        self.P = U.shape[1]
        self.Q = V.shape[1]
        self.U = U.copy()
        self.V = V.copy()
        self.block_sums = np.zeros((self.P, self.Q), dtype=float)
        self.block_counts = np.zeros((self.P, self.Q), dtype=float)
        self._rebuild()

    def _rebuild(self):
        self.block_sums.fill(0.0)
        self.block_counts.fill(0.0)
        ra = np.argmax(self.U, axis=1)
        ca = np.argmax(self.V, axis=1)
        for p in range(self.P):
            rmask = ra == p
            if not rmask.any():
                continue
            Xp = self.X[rmask]
            for q in range(self.Q):
                cmask = ca == q
                if not cmask.any():
                    continue
                blk = Xp[:, cmask]
                self.block_sums[p, q] += blk.sum()
                self.block_counts[p, q] += blk.size

    def current_C(self):
        C = self.block_sums / np.maximum(self.block_counts, 1e-12)
        C[self.block_counts == 0] = 0.0
        return C

    def update_rows(self, rows, U_new_rows, V, Q):
        U_old_rows = self.U[rows]
        ra_old = np.argmax(U_old_rows, axis=1)
        ra_new = np.argmax(U_new_rows, axis=1)
        ca = np.argmax(V, axis=1)
        for k, i in enumerate(rows):
            p_old = ra_old[k]
            p_new = ra_new[k]
            if p_old == p_new:
                continue
            xi = self.X[i]
            for q in range(Q):
                cmask = ca == q
                if cmask.any():
                    vals = xi[cmask]
                    self.block_sums[p_old, q] -= vals.sum()
                    self.block_counts[p_old, q] -= vals.size
                    self.block_sums[p_new, q] += vals.sum()
                    self.block_counts[p_new, q] += vals.size
        self.U[rows] = U_new_rows

    def update_cols(self, cols, V_new_cols, U, P):
        V_old_cols = self.V[cols]
        ca_old = np.argmax(V_old_cols, axis=1)
        ca_new = np.argmax(V_new_cols, axis=1)
        ra = np.argmax(U, axis=1)
        for k, j in enumerate(cols):
            q_old = ca_old[k]
            q_new = ca_new[k]
            if q_old == q_new:
                continue
            xj = self.X[:, j]
            for p in range(P):
                rmask = ra == p
                if rmask.any():
                    vals = xj[rmask]
                    self.block_sums[p, q_old] -= vals.sum()
                    self.block_counts[p, q_old] -= vals.size
                    self.block_sums[p, q_new] += vals.sum()
                    self.block_counts[p, q_new] += vals.size
        self.V[cols] = V_new_cols


# ================= Estimator =================
class TiedGDKM(BaseEstimator, BiclusterMixin):
    """
    GDKM-Tied with balance-aware tie-breaker and warm-start from labels
    with optional history.

    Parameters
    ----------
    n_row_clusters : int
    n_col_clusters : int
    norm : {'l2','l1','huber','mav_ratio'}, default='l2'
        Distance norm to use. Options: "l1", "l2", "huber", "mav_ratio".
        - "l1": L1 norm (Manhattan distance)
        - "l2": L2 norm (Euclidean distance)
        - "huber": Huber loss (robust to outliers)
        - "mav_ratio": Mean Absolute Value ratio (MAE / MAV)
    huber_delta : float, default=1.0
        Huber loss parameter.
        Ignored if norm != 'huber'.
    alpha_empty : float, default=0.0
        Penalty per empty row/col cluster.
    alpha_var : float, default=0.0
        Penalty on variance of cluster sizes (rows + cols).
    solver : {'batch','online'}, default='batch'
        Optimization solver to use.
        - 'batch': Full batch (original GDKM-Tied)
        - 'online': Online mini-batch (experimental)
    row_batch, col_batch : int, default=256
        Batch sizes for online solver.
        Ignored if solver != 'online'.
    tie_breaker : {'first','random','balance'}, default='first'
        How to resolve assignment ties.
        - 'first':  First index wins.
        - 'random': Randomly choose one.
        - 'balance': Balance-aware tiebreaker.
    balance_power : float, default=1.0
        Strength of balance preference when tie_breaker='balance'.
    row_labels0, col_labels0 : array-like or None
        Optional warm-start labels. Use -1 for 'random' on that sample/feature.
        Ignored if U0/V0 provided.
    keep_history : bool, default=False
        Whether to keep the objective history.
    """

    def __init__(
        self,
        n_row_clusters,
        n_col_clusters,
        *,
        norm="l2",
        stat="mean",
        huber_delta=1.0,
        alpha_empty=0.0,
        alpha_var=0.0,
        solver="batch",
        max_iter=100,
        tol=1e-4,
        random_state=None,
        keep_history=False,
        track_objective="l2",
        use_trace=False,
        row_batch=256,
        col_batch=256,
        online_steps=200,
        tie_breaker="first",
        balance_power=1.0,
        row_labels0=None,
        col_labels0=None,
    ):
        self.n_row_clusters = int(n_row_clusters)
        self.n_col_clusters = int(n_col_clusters)
        self.norm = norm
        self.stat = stat
        self.huber_delta = huber_delta
        self.alpha_empty = alpha_empty
        self.alpha_var = alpha_var
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.keep_history = keep_history
        self.track_objective = track_objective
        self.use_trace = use_trace
        self.row_batch = row_batch
        self.col_batch = col_batch
        self.online_steps = online_steps
        self.tie_breaker = tie_breaker
        self.balance_power = balance_power
        self.row_labels0 = row_labels0
        self.col_labels0 = col_labels0

    def fit(self, X, y=None, U0=None, V0=None):
        X = check_array(X, accept_sparse=False, dtype=float, ensure_2d=True)
        rng = np.random.default_rng(self.random_state)
        I, J = X.shape
        P, Q = self.n_row_clusters, self.n_col_clusters

        # ----- Warm start logic -----
        if U0 is not None:
            U = np.asarray(U0, dtype=int).copy()
        elif self.row_labels0 is not None:
            U = _one_hot_from_labels(self.row_labels0, P, rng)
        else:
            U = np.zeros((I, P), dtype=int)
            U[np.arange(I), rng.integers(0, P, size=I)] = 1

        if V0 is not None:
            V = np.asarray(V0, dtype=int).copy()
        elif self.col_labels0 is not None:
            V = _one_hot_from_labels(self.col_labels0, Q, rng)
        else:
            V = np.zeros((J, Q), dtype=int)
            V[np.arange(J), rng.integers(0, Q, size=J)] = 1

        history = []
        n_iter = 0

        if self.solver == "batch":
            prev_obj = None
            for t in range(self.max_iter):
                n_iter = t + 1
                C = _update_C_tied(X, U, V, stat=self.stat)

                # counts for balance tiebreak
                row_counts = cluster_counts(U)  # (P,)
                U_new = _update_U_tied(
                    X,
                    C,
                    V,
                    norm=self.norm,
                    huber_delta=self.huber_delta,
                    tie_breaker=self.tie_breaker,
                    balance_power=self.balance_power,
                    counts=row_counts,
                    rng=rng,
                )

                col_counts = cluster_counts(V)  # (Q,)
                V_new = _update_V_tied(
                    X,
                    U_new,
                    C,
                    norm=self.norm,
                    huber_delta=self.huber_delta,
                    tie_breaker=self.tie_breaker,
                    balance_power=self.balance_power,
                    counts=col_counts,
                    rng=rng,
                )

                # objective + penalties
                if self.track_objective == "l2":
                    base = (
                        l2_objective_trace(X, U_new, C, V_new)
                        if self.use_trace
                        else l2_objective(X, U_new, C, V_new)
                    )
                elif self.track_objective == "l1":
                    base = l1_objective(X, U_new, C, V_new)
                elif self.track_objective == "mav":
                    base = mav_objective(X, U_new, C, V_new)
                else:
                    raise ValueError("track_objective must be 'l2','l1','mav'")
                obj = base + cluster_balance_penalty(
                    U_new, V_new, alpha_empty=self.alpha_empty, alpha_var=self.alpha_var
                )
                if self.keep_history:
                    history.append(obj)

                # stop conditions
                if (U_new == U).all() and (V_new == V).all():
                    U, V = U_new, V_new
                    break
                if prev_obj is not None:
                    rel_impr = (prev_obj - obj) / max(prev_obj, 1e-12)
                    if 0 <= rel_impr < self.tol:
                        U, V = U_new, V_new
                        break
                U, V = U_new, V_new
                prev_obj = obj

            C = _update_C_tied(X, U, V, stat=self.stat)

        elif self.solver == "online":
            state = _OnlineState(X, U, V)
            Iidx = np.arange(I)
            Jidx = np.arange(J)
            for t in range(self.online_steps):
                n_iter = t + 1

                # --- row mini-batch with tiebreak based on current counts
                br = rng.choice(Iidx, size=min(self.row_batch, I), replace=False)
                C = state.current_C()
                row_counts = cluster_counts(state.U)
                U_new_rows = _update_U_tied(
                    X[br],
                    C,
                    state.V,
                    norm=self.norm,
                    huber_delta=self.huber_delta,
                    tie_breaker=self.tie_breaker,
                    balance_power=self.balance_power,
                    counts=row_counts,
                    rng=rng,
                )
                state.update_rows(br, U_new_rows, state.V, Q)

                # --- column mini-batch
                bc = rng.choice(Jidx, size=min(self.col_batch, J), replace=False)
                C = state.current_C()
                col_counts = cluster_counts(state.V)
                V_new_cols = _update_V_tied(
                    X[:, bc],
                    state.U,
                    C,
                    norm=self.norm,
                    huber_delta=self.huber_delta,
                    tie_breaker=self.tie_breaker,
                    balance_power=self.balance_power,
                    counts=col_counts,
                    rng=rng,
                )
                state.update_cols(bc, V_new_cols, state.U, P)

                # track objective
                C = state.current_C()
                if self.track_objective == "l2":
                    base = l2_objective(X, state.U, C, state.V)
                elif self.track_objective == "l1":
                    base = l1_objective(X, state.U, C, state.V)
                elif self.track_objective == "mav":
                    base = mav_objective(X, state.U, C, state.V)
                obj = base + cluster_balance_penalty(
                    state.U,
                    state.V,
                    alpha_empty=self.alpha_empty,
                    alpha_var=self.alpha_var,
                )
                if self.keep_history:
                    history.append(obj)

            U, V, C = state.U, state.V, state.current_C()

        else:
            raise ValueError("solver must be 'batch' or 'online'")

        # finalize
        self.U_ = U
        self.V_ = V
        self.C_ = C
        self.row_labels_ = np.argmax(U, axis=1)
        self.column_labels_ = np.argmax(V, axis=1)
        self.objective_history_ = history
        self.n_iter_ = n_iter
        self.name_ = "GDKM-Tied"
        return self

    # utilities
    def get_indices(self):
        return self.row_labels_, self.column_labels_

    def reconstruct(self):
        return self.U_ @ self.C_ @ self.V_.T

    def score(self, X):
        X = check_array(X, accept_sparse=False, dtype=float, ensure_2d=True)
        return -l2_objective(X, self.U_, self.C_, self.V_)

    def get_params(self, deep=False):
        return {
            "n_row_clusters": self.n_row_clusters,
            "n_col_clusters": self.n_col_clusters,
            "norm": self.norm,
            "stat": self.stat,
            "huber_delta": self.huber_delta,
            "alpha_empty": self.alpha_empty,
            "alpha_var": self.alpha_var,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
            "track_objective": self.track_objective,
            "use_trace": self.use_trace,
            "row_batch": self.row_batch,
            "col_batch": self.col_batch,
            "online_steps": self.online_steps,
        }

    def get_row_clusters(self):
        """Return row cluster labels."""
        return self.row_labels_

    def get_column_clusters(self):
        """Return column cluster labels."""
        return self.column_labels_

    def get_biclusters(self):
        """
        Return boolean indicator matrices (rows, columns) for each bicluster.
        Returns all (p,q) combinations with p in [0..P-1], q in [0..Q-1].

        Returns
        -------
        rows : ndarray of shape (n_biclusters, n_samples)
            Boolean array indicating which samples belong to each bicluster.
        cols : ndarray of shape (n_biclusters, n_features)
            Boolean array indicating which features belong to each bicluster.
        """
        if not hasattr(self, "U_") or not hasattr(self, "V_"):
            raise AttributeError("Must call fit() before get_biclusters().")

        P = self.U_.shape[1]
        Q = self.V_.shape[1]
        rows_list, cols_list = [], []

        for p in range(P):
            row_mask = self.U_[:, p].astype(bool)
            for q in range(Q):
                col_mask = self.V_[:, q].astype(bool)
                if row_mask.any() and col_mask.any():
                    rows_list.append(row_mask)
                    cols_list.append(col_mask)

        return np.array(rows_list), np.array(cols_list)

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """String representation of the TiedGDKM object."""
        params = []
        params.append(f"n_row_clusters={self.n_row_clusters}")
        params.append(f"n_col_clusters={self.n_col_clusters}")
        params.append(f"norm='{self.norm}'")
        params.append(f"stat='{self.stat}'")
        params.append(f"solver='{self.solver}'")
        params.append(f"max_iter={self.max_iter}")
        params.append(f"tol={self.tol}")
        params.append(f"random_state={self.random_state}")

        # TiedGDKM-specific parameters
        if self.tie_breaker != "first":
            params.append(f"tie_breaker='{self.tie_breaker}'")
        if self.balance_power != 1.0:
            params.append(f"balance_power={self.balance_power}")
        if self.row_labels0 is not None:
            params.append("row_labels0=<provided>")
        if self.col_labels0 is not None:
            params.append("col_labels0=<provided>")

        # Optional parameters
        if self.huber_delta != 1.0:
            params.append(f"huber_delta={self.huber_delta}")
        if self.alpha_empty != 0.0:
            params.append(f"alpha_empty={self.alpha_empty}")
        if self.alpha_var != 0.0:
            params.append(f"alpha_var={self.alpha_var}")
        if self.track_objective != "l2":
            params.append(f"track_objective='{self.track_objective}'")
        if self.use_trace:
            params.append(f"use_trace={self.use_trace}")
        if self.solver == "online":
            params.append(f"row_batch={self.row_batch}")
            params.append(f"col_batch={self.col_batch}")
            params.append(f"online_steps={self.online_steps}")

        s = f"TiedGDKM({', '.join(params)})"
        if N_CHAR_MAX is not None and N_CHAR_MAX > 0 and len(s) > N_CHAR_MAX:
            return s[: N_CHAR_MAX - 3] + "..."
        return s


# class SolverGDKM(BaseEstimator, BiclusterMixin):
#     """
#     GDKM-Tied: piecewise-constant biclustering with hard row/col assignments.

#     Parameters
#     ----------
#     n_row_clusters : int
#     n_col_clusters : int
#     norm : {'l2','l1','huber','mav_ratio'}, default='l2'
#     stat : {'mean','median'}, default='mean'
#         How to update C per block: 'mean' (L2-optimal) or 'median' (L1-optimal).
#     huber_delta : float, default=1.0
#     alpha_empty : float, default=0.0
#         Penalty per empty row/col cluster.
#     alpha_var : float, default=0.0
#         Penalty on variance of cluster sizes (rows + cols).
#     solver : {'batch','online'}, default='batch'
#     max_iter : int, default=100
#     tol : float, default=1e-4
#     random_state : int or None
#     track_objective : {'l2','l1','mav'}, default='l2'
#         Which objective to log in history (balance penalties added on top).
#     use_trace : bool, default=False
#         Use trace form for L2 objective (memory-light).

#     Attributes (after fit)
#     ----------------------
#     U_ : (I, P) int one-hot
#     V_ : (J, Q) int one-hot
#     C_ : (P, Q) float
#     row_labels_ : (I,) int
#     column_labels_ : (J,) int
#     objective_history_ : list of floats
#     n_iter_ : int
#     name_ : str == "GDKM-Tied"
#     """

#     def __init__(
#         self,
#         n_row_clusters,
#         n_col_clusters,
#         *,
#         norm="l2",
#         stat="mean",
#         huber_delta=1.0,
#         alpha_empty=0.0,
#         alpha_var=0.0,
#         solver="batch",
#         max_iter=100,
#         tol=1e-4,
#         random_state=None,
#         track_objective="l2",
#         use_trace=False,
#         row_batch=256,
#         col_batch=256,
#         online_steps=200,
#     ):
#         self.n_row_clusters = int(n_row_clusters)
#         self.n_col_clusters = int(n_col_clusters)
#         self.norm = norm
#         self.stat = stat
#         self.huber_delta = huber_delta
#         self.alpha_empty = alpha_empty
#         self.alpha_var = alpha_var
#         self.solver = solver
#         self.max_iter = max_iter
#         self.tol = tol
#         self.random_state = random_state
#         self.track_objective = track_objective
#         self.use_trace = use_trace
#         # online params
#         self.row_batch = row_batch
#         self.col_batch = col_batch
#         self.online_steps = online_steps

#     # -------- API --------
#     def fit(self, X, y=None, U0=None, V0=None):
#         X = check_array(X, accept_sparse=False, dtype=float, ensure_2d=True)
#         rng = np.random.default_rng(self.random_state)
#         I, J = X.shape
#         P, Q = self.n_row_clusters, self.n_col_clusters

#         # init U, V
#         if U0 is None:
#             U = np.zeros((I, P), dtype=int)
#             U[np.arange(I), rng.integers(0, P, size=I)] = 1
#         else:
#             U = np.asarray(U0, dtype=int).copy()
#         if V0 is None:
#             V = np.zeros((J, Q), dtype=int)
#             V[np.arange(J), rng.integers(0, Q, size=J)] = 1
#         else:
#             V = np.asarray(V0, dtype=int).copy()

#         history = []
#         n_iter = 0

#         if self.solver == "batch":
#             prev_obj = None
#             for t in range(self.max_iter):
#                 n_iter = t + 1
#                 C = _update_C_tied(X, U, V, stat=self.stat)
#                 U_new = _update_U_tied(
#                     X, C, V, norm=self.norm, huber_delta=self.huber_delta
#                 )
#                 V_new = _update_V_tied(
#                     X, U_new, C, norm=self.norm, huber_delta=self.huber_delta
#                 )

#                 # compute objective to track
#                 if self.track_objective == "l2":
#                     base = (
#                         l2_objective_trace(X, U_new, C, V_new)
#                         if self.use_trace
#                         else l2_objective(X, U_new, C, V_new)
#                     )
#                 elif self.track_objective == "l1":
#                     base = l1_objective(X, U_new, C, V_new)
#                 elif self.track_objective == "mav":
#                     base = mav_objective(X, U_new, C, V_new)
#                 else:
#                     raise ValueError("track_objective must be 'l2','l1','mav'")

#                 obj = base + cluster_balance_penalty(
#                     U_new, V_new, alpha_empty=self.alpha_empty, alpha_var=self.alpha_var
#                 )
#                 history.append(obj)

#                 # stop if assignments stable or small relative improvement
#                 if (U_new == U).all() and (V_new == V).all():
#                     U, V = U_new, V_new
#                     break
#                 if prev_obj is not None:
#                     rel_impr = (prev_obj - obj) / max(prev_obj, 1e-12)
#                     if 0 <= rel_impr < self.tol:
#                         U, V = U_new, V_new
#                         break

#                 U, V = U_new, V_new
#                 prev_obj = obj

#             C = _update_C_tied(X, U, V, stat=self.stat)

#         elif self.solver == "online":
#             # build state and alternate row/col mini-batches
#             state = _OnlineState(X, U, V)
#             Iidx = np.arange(I)
#             Jidx = np.arange(J)
#             for t in range(self.online_steps):
#                 n_iter = t + 1
#                 # row batch
#                 br = rng.choice(Iidx, size=min(self.row_batch, I), replace=False)
#                 C = state.current_C()
#                 U_new_rows = _update_U_tied(
#                     X[br], C, state.V, norm=self.norm, huber_delta=self.huber_delta
#                 )
#                 state.update_rows(br, U_new_rows, state.V, self.n_col_clusters)

#                 # col batch
#                 bc = rng.choice(Jidx, size=min(self.col_batch, J), replace=False)
#                 C = state.current_C()
#                 V_new_cols = _update_V_tied(
#                     X[:, bc], state.U, C, norm=self.norm, huber_delta=self.huber_delta
#                 )
#                 state.update_cols(bc, V_new_cols, state.U, self.n_row_clusters)

#                 # track objective periodically (every step here)
#                 C = state.current_C()
#                 if self.track_objective == "l2":
#                     base = l2_objective(X, state.U, C, state.V)
#                 elif self.track_objective == "l1":
#                     base = l1_objective(X, state.U, C, state.V)
#                 elif self.track_objective == "mav":
#                     base = mav_objective(X, state.U, C, state.V)
#                 obj = base + cluster_balance_penalty(
#                     state.U,
#                     state.V,
#                     alpha_empty=self.alpha_empty,
#                     alpha_var=self.alpha_var,
#                 )
#                 history.append(obj)

#             U, V, C = state.U, state.V, state.current_C()

#         else:
#             raise ValueError("solver must be 'batch' or 'online'")

#         # finalize attributes
#         self.U_ = U
#         self.V_ = V
#         self.C_ = C
#         self.row_labels_ = np.argmax(U, axis=1)
#         self.column_labels_ = np.argmax(V, axis=1)
#         self.objective_history_ = history
#         self.n_iter_ = n_iter
#         self.name_ = "GDKM-Tied"
#         return self

#     # --- utilities ---
#     def get_indices(self):
#         """Bicluster indices in sklearn convention."""
#         return self.row_labels_, self.column_labels_

#     def reconstruct(self):
#         """Return U C V^T."""
#         return self.U_ @ self.C_ @ self.V_.T

#     def score(self, X):
#         """Negative L2 reconstruction error (higher is better)."""
#         X = check_array(X, accept_sparse=False, dtype=float, ensure_2d=True)
#         return -l2_objective(X, self.U_, self.C_, self.V_)
