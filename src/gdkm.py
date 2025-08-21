# Generalized Double K-Means (GDKM) implementation
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.base import BaseEstimator, BiclusterMixin

import logging

logger = logging.getLogger(__name__)


def _get_stats(norm: str):
    if norm == "l1":
        return "median"
    else:
        return "mean"


class GeneralizedDoubleKMeans(BaseEstimator, BiclusterMixin):
    def __init__(
        self,
        n_row_clusters=3,
        n_col_clusters=None,  # NEW: number of global item clusters (when tie_columns=True)
        n_col_clusters_list=None,  # keep for per-row-cluster columns (tie_columns=False)
        tie_columns: bool = True,  # NEW: global V if True; your original V_list if False
        max_iter=100,
        tol=1e-4,
        random_state=None,
        norm="l2",
        ensure_min_size: int = 1,  # NEW: optionally enforce min size per cluster
        huber_delta: float = 1.0,  # NEW: Huber loss parameter
    ):
        """
        Initialize GeneralizedDoubleKMeans.

        Parameters
        ----------
        norm : str, default="l2"
            Distance norm to use. Options: "l1", "l2", "huber", "mav_ratio".
            - "l1": L1 norm (Manhattan distance)
            - "l2": L2 norm (Euclidean distance)
            - "huber": Huber loss (robust to outliers)
            - "mav_ratio": Mean Absolute Value ratio (MAE / MAV)
        """
        self.n_row_clusters = n_row_clusters
        self.n_col_clusters = n_col_clusters
        self.n_col_clusters_list = n_col_clusters_list
        self.tie_columns = tie_columns
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.norm = norm
        self.stats = _get_stats(norm)
        logger.info(f"Initialized with norm={norm}, stats={self.stats}")
        self.ensure_min_size = ensure_min_size
        self.huber_delta = huber_delta

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """String representation of the GeneralizedDoubleKMeans object."""
        params = []
        params.append(f"n_row_clusters={self.n_row_clusters}")

        if self.tie_columns:
            params.append(f"n_col_clusters={self.n_col_clusters}")
            params.append(f"tie_columns={self.tie_columns}")
        else:
            params.append(f"n_col_clusters_list={self.n_col_clusters_list}")
            params.append(f"tie_columns={self.tie_columns}")

        params.append(f"max_iter={self.max_iter}")
        params.append(f"tol={self.tol}")
        params.append(f"random_state={self.random_state}")
        params.append(f"norm='{self.norm}'")
        params.append(f"ensure_min_size={self.ensure_min_size}")
        params.append(f"huber_delta={self.huber_delta}")
        params.append(f"stats='{self.stats}'")

        s = f"GeneralizedDoubleKMeans({', '.join(params)})"
        if N_CHAR_MAX is not None and N_CHAR_MAX > 0 and len(s) > N_CHAR_MAX:
            return s[: N_CHAR_MAX - 3] + "..."
        return s

    def fit(self, X):
        X = np.asarray(X)
        I, J = X.shape
        rng = np.random.default_rng(self.random_state)

        if self.tie_columns:
            if self.n_col_clusters is None:
                raise ValueError("Set n_col_clusters when tie_columns=True")
            # ===== TIED COLUMNS PATH =====
            U, V = _init_UV_tied(I, J, self.n_row_clusters, self.n_col_clusters, rng)
            prev_obj = np.inf
            for _ in range(self.max_iter):
                C = _update_C_tied(X, U, V, stat=self.stats)  # C: (P, Q)
                U = _update_U_tied(
                    X, C, V, self.norm, huber_delta=self.huber_delta
                )  # U: (I, P) one-hot
                V = _update_V_tied(X, U, C, self.norm)  # V: (J, Q) one-hot
                _enforce_min_size(U, axis=0, min_k=self.ensure_min_size, rng=rng)
                _enforce_min_size(V, axis=0, min_k=self.ensure_min_size, rng=rng)
                obj = _loss_tied(X, U, C, V, self.norm)
                if abs(prev_obj - obj) < self.tol:
                    break
                prev_obj = obj

            self.U_ = U
            self.V_ = V
            self.C_ = C
            self.loss_ = prev_obj
            self.row_labels_ = np.argmax(U, axis=1)
            self.column_labels_ = np.argmax(V, axis=1)  # GLOBAL item labels (0..Q-1)
            # biclusters_ property is handled by BiclusterMixin
            return self

        else:
            # ===== YOUR ORIGINAL PER-ROW-CLUSTER PATH =====
            if self.n_col_clusters_list is None:
                # default: same Q for each row cluster if n_col_clusters given
                if self.n_col_clusters is None:
                    raise ValueError("Provide n_col_clusters_list or n_col_clusters")
                self.n_col_clusters_list = [self.n_col_clusters] * self.n_row_clusters

            self.U_, self.V_list_, self.C_blocks_, self.loss_ = (
                generalized_double_kmeans(
                    X,
                    P=self.n_row_clusters,
                    Q_list=self.n_col_clusters_list,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state,
                    norm=self.norm,
                )
            )
            self.row_labels_ = np.argmax(self.U_, axis=1)

            # Provide a GLOBAL item label if you need one:
            # Here we choose the (p,q) with maximum support across rows assigned to p.
            self.column_labels_ = _globalize_item_labels_from_Vlist(
                self.V_list_, self.U_
            )  # labels in 0..sum(Qp)-1 (composite)

            # biclusters_ property is handled by BiclusterMixin
            return self

    def get_row_clusters(self):
        return self.row_labels_

    def get_column_clusters(self):
        return getattr(self, "column_labels_", None)

    def get_biclusters(self):
        """
        Return boolean indicator matrices (rows, columns) for each bicluster.
        If tie_columns=True, biclusters are all (p,q) with p in [0..P-1], q in [0..Q-1].
        If tie_columns=False, biclusters follow your original (p, q in Qp) blocks.
        """
        if self.tie_columns:
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
        else:
            # original behavior using V_list_
            if not hasattr(self, "U_") or not hasattr(self, "V_list_"):
                raise AttributeError("Must call fit() before get_biclusters().")
            rows, cols = [], []
            P = self.U_.shape[1]
            for p in range(P):
                row_mask = self.U_[:, p] == 1
                Vp = self.V_list_[p]  # (J, Qp)
                for q in range(Vp.shape[1]):
                    col_mask = Vp[:, q] == 1
                    if np.any(row_mask) and np.any(col_mask):
                        rows.append(row_mask)
                        cols.append(col_mask)
            return np.array(rows), np.array(cols)


def _init_UV_tied(I, J, P, Q, rng):
    U = np.zeros((I, P), dtype=int)
    U[np.arange(I), rng.integers(P, size=I)] = 1
    V = np.zeros((J, Q), dtype=int)
    V[np.arange(J), rng.integers(Q, size=J)] = 1
    _enforce_min_size(U, axis=0, min_k=1, rng=rng)
    _enforce_min_size(V, axis=0, min_k=1, rng=rng)
    return U, V


def _enforce_min_size(M, axis, min_k, rng):
    """Ensure every cluster (columns if axis=0; rows if axis=1) has at least min_k members."""
    if min_k <= 0:
        return
    if axis == 0:
        counts = M.sum(axis=0)
        for k in np.where(counts < min_k)[0]:
            # randomly reassign some rows/cols into k
            idx = rng.choice(M.shape[0], size=(min_k - counts[k]), replace=False)
            M[idx] = 0
            M[idx, k] = 1
    else:
        counts = M.sum(axis=1)
        for i in np.where(counts == 0)[0]:
            k = rng.integers(M.shape[1])
            M[i, k] = 1


def _update_C_tied(X, U, V, *, stat="mean"):
    """
    Vectorized C update (P x Q) for one-hot U (I x P), V (J x Q).
    stat='mean' (L2-optimal) or 'median' (L1-optimal).
    """
    I, J = X.shape
    P = U.shape[1]
    Q = V.shape[1]

    # Compute block sums & counts with matrix multiplies:
    # sums[p,q] = sum_{i in p, j in q} X[i,j]
    sums = U.T @ X @ V  # (P, Q)
    counts = (U.T @ np.ones((I, 1))) @ (np.ones((1, J)) @ V)  # (P, Q)

    if stat == "mean":
        C = sums / np.maximum(counts, 1e-12)
        # Optional: keep zeros when truly empty
        C[counts == 0] = 0.0
        return C

    elif stat == "median":
        # True L1-optimal block statistics: per-block medians
        # This path falls back to masked loops only over non-empty blocks.
        C = np.zeros((P, Q), dtype=X.dtype)
        row_assign = np.argmax(U, axis=1)  # (I,)
        col_assign = np.argmax(V, axis=1)  # (J,)
        for p in range(P):
            rmask = row_assign == p
            if not rmask.any():
                continue
            Xp = X[rmask]  # (n_p, J)
            for q in range(Q):
                cmask = col_assign == q
                if not cmask.any():
                    continue
                block = Xp[:, cmask]
                if block.size:
                    C[p, q] = np.median(block)
                # else leave at 0.0
        return C

    else:
        raise ValueError("stat must be 'mean' or 'median'")


# def _update_C_tied(X, U, V):
#     """C = argmin ||X - U C V^T|| given one-hot U,V → C[p,q] = mean of block (p,q)."""
#     P = U.shape[1]
#     Q = V.shape[1]
#     C = np.zeros((P, Q), dtype=float)
#     for p in range(P):
#         rmask = U[:, p] == 1
#         if not rmask.any():
#             continue
#         Xp = X[rmask]  # (n_p, J)
#         for q in range(Q):
#             cmask = V[:, q] == 1
#             if not cmask.any():
#                 continue
#             block = Xp[:, cmask]
#             C[p, q] = block.mean() if block.size else 0.0
#     return C


def _update_U_tied(X, C, V, norm="l2", huber_delta=1.0):
    """
    Vectorized row assignment: U[i,:] = one-hot argmin_p loss(x_i, prototype_p).
    """
    I, J = X.shape
    P, Q = C.shape

    # prototypes[p, j] = C[p, q(j)]
    prototypes = C @ V.T  # (P, J)

    # Broadcast to (I, P, J): compare each row to each prototype row
    # X_exp[i, p, j] = X[i, j]
    X_exp = X[:, None, :]  # (I, 1, J)
    P_exp = prototypes[None, :, :]  # (1, P, J)
    R = X_exp - P_exp  # residuals (I, P, J)

    if norm == "l2":
        errs = np.sum(R * R, axis=2)  # (I, P)
    elif norm == "l1":
        errs = np.sum(np.abs(R), axis=2)  # (I, P)
    elif norm == "huber":
        errs = np.sum(huber_loss(R, delta=huber_delta), axis=2)
    elif norm == "mav_ratio":
        mae = np.sum(np.abs(R), axis=2)  # per (i,p)
        mav = np.sum(np.abs(X), axis=1, keepdims=True)  # (I,1)
        errs = mae / np.maximum(mav, 1e-12)
    else:
        raise ValueError("Unsupported norm: 'l1', 'l2', 'huber', or 'mav_ratio'")

    U = np.zeros((I, P), dtype=int)
    U[np.arange(I), np.argmin(errs, axis=1)] = 1
    return U


# def _update_U_tied(X, C, V, norm="l2"):
#     """Assign each row i to p that minimizes ||x_i - C[p,:] @ V^T||."""
#     I, J = X.shape
#     P, Q = C.shape
#     prototypes = C @ V.T  # (P, J)
#     errs = np.zeros((I, P))
#     if norm == "l2":
#         for p in range(P):
#             diff = X - prototypes[p]  # (I, J)
#             errs[:, p] = np.sum(diff * diff, axis=1)
#     elif norm == "l1":
#         for p in range(P):
#             errs[:, p] = np.sum(np.abs(X - prototypes[p]), axis=1)
#     elif norm == "huber":
#         for p in range(P):
#             errs[:, p] = np.sum(huber_loss(X - prototypes[p]), axis=1)
#     elif norm == "mav_ratio":
#         for p in range(P):
#             diff = X - prototypes[p]  # (I, J)
#             mae = np.sum(np.abs(diff), axis=1)  # mean absolute error per row
#             mav = np.sum(np.abs(X), axis=1)  # mean absolute value per row
#             errs[:, p] = mae / np.maximum(mav, 1e-12)  # MAV ratio per row
#     else:
#         raise ValueError("Unsupported norm: use 'l1', 'l2', 'huber', or 'mav_ratio'")
#     U = np.zeros((I, P), dtype=int)
#     U[np.arange(I), np.argmin(errs, axis=1)] = 1
#     return U


def _update_V_tied(X, U, C, norm="l2", huber_delta=1.0):
    """
    Vectorized column assignment: V[j,:] = one-hot argmin_q loss(x_:j, basis_q).
    """
    I, J = X.shape
    P, Q = C.shape

    # bases[i, q] = C[p(i), q]
    bases = U @ C  # (I, Q)

    # Broadcast to (J, Q, I) by working row-major then swapping axes:
    # Compare each column to each basis column across rows.
    X_col = X.T[:, None, :]  # (J, 1, I)
    B_col = bases.T[None, :, :]  # (1, Q, I)
    R = X_col - B_col  # residuals (J, Q, I)

    if norm == "l2":
        errs = np.sum(R * R, axis=2)  # (J, Q)
    elif norm == "l1":
        errs = np.sum(np.abs(R), axis=2)  # (J, Q)
    elif norm == "huber":
        errs = np.sum(huber_loss(R, delta=huber_delta), axis=2)
    elif norm == "mav_ratio":
        mae = np.sum(np.abs(R), axis=2)  # per (j,q)
        mav = np.sum(np.abs(X), axis=0, keepdims=True).T  # (J,1)
        errs = mae / np.maximum(mav, 1e-12)
    else:
        raise ValueError("Unsupported norm: 'l1', 'l2', 'huber', or 'mav_ratio'")

    V = np.zeros((J, Q), dtype=int)
    V[np.arange(J), np.argmin(errs, axis=1)] = 1
    return V


# def _update_V_tied(X, U, C, norm="l2"):
#     """Assign each column j to q that minimizes ||x_:j - U @ C[:, q]||."""
#     I, J = X.shape
#     P, Q = C.shape
#     bases = U @ C  # (I, Q)
#     errs = np.zeros((J, Q))
#     if norm == "l2":
#         for q in range(Q):
#             diff = X - bases[:, [q]]  # (I, J)
#             errs[:, q] = np.sum((diff * diff), axis=0)
#     elif norm == "l1":
#         for q in range(Q):
#             errs[:, q] = np.sum(np.abs(X - bases[:, [q]]), axis=0)
#     elif norm == "huber":
#         for q in range(Q):
#             errs[:, q] = np.sum(huber_loss(X - bases[:, [q]]), axis=0)
#     elif norm == "mav_ratio":
#         for q in range(Q):
#             diff = X - bases[:, [q]]  # (I, J)
#             mae = np.sum(np.abs(diff), axis=0)  # mean absolute error per column
#             mav = np.sum(np.abs(X), axis=0)  # mean absolute value per column
#             errs[:, q] = mae / np.maximum(mav, 1e-12)  # MAV ratio per column
#     else:
#         raise ValueError("Unsupported norm: use 'l1', 'l2', 'huber', or 'mav_ratio'")
#     V = np.zeros((J, Q), dtype=int)
#     V[np.arange(J), np.argmin(errs, axis=1)] = 1
#     return V


def _loss_tied(X, U, C, V, norm="l2"):
    recon = (U @ C) @ V.T
    if norm == "l2":
        return float(np.sum((X - recon) ** 2))
    if norm == "l1":
        return float(np.sum(np.abs(X - recon)))
    if norm == "huber":
        return float(np.sum(huber_loss(X - recon)))
    if norm == "mav_ratio":
        diff = X - recon
        mae = np.sum(np.abs(diff))
        mav = np.sum(np.abs(X))
        return float(mae / max(mav, 1e-12))  # MAV ratio as loss
    raise ValueError("Unsupported norm: use 'l1', 'l2', 'huber', or 'mav_ratio'")


def _globalize_item_labels_from_Vlist(V_list, U):
    """
    Map per-row-cluster V_list to a single global label per column.
    We choose the (p,q) with the largest count of rows assigned to p (weighted support).
    Labels are in [0 .. sum(Qp)-1], grouped by p's offsets.
    """
    P = U.shape[1]
    Q_offsets = np.cumsum([0] + [V_list[p].shape[1] for p in range(P)])  # offsets
    J = V_list[0].shape[0]
    labels = np.zeros(J, dtype=int)
    support = np.zeros((J, Q_offsets[-1]), dtype=int)
    rows_per_p = U.sum(axis=0)  # support weight per row-cluster
    for p in range(P):
        Vp = V_list[p]  # (J, Qp)
        off = Q_offsets[p]
        # count rows in p as support (simple weighting)
        w = int(rows_per_p[p])
        support[np.arange(J)[:, None], off + np.argmax(Vp, axis=1)[:, None]] += w
    labels = np.argmax(support, axis=1)
    return labels


def huber_loss(diff, delta=1.0):
    """
    Compute Huber loss for a difference array.

    Parameters:
    - diff: np.ndarray, difference between prediction and actual
    - delta: threshold between L2 and L1 behavior

    Returns:
    - np.ndarray of same shape as diff
    """
    abs_diff = np.abs(diff)
    quadratic = np.minimum(abs_diff, delta)
    linear = abs_diff - quadratic
    return 0.5 * quadratic**2 + delta * linear


def initialize_partitions(I, J, P, Q_list, random_state=None):
    """
    Randomly initialize row and column cluster assignments.

    Parameters:
    - I: number of rows in the data matrix
    - J: number of columns in the data matrix
    - P: number of row clusters
    - Q_list: list of number of column clusters per row cluster (length P)
    - random_state: optional seed for reproducibility

    Returns:
    - U: binary matrix of shape (I, P), row-cluster assignments
    - V_list: list of binary matrices (J, Qp), one for each row cluster p
    """
    rng = np.random.default_rng(random_state)

    # Initialize U: assign each row to a random row cluster
    U = np.zeros((I, P), dtype=int)
    for i in range(I):
        U[i, rng.integers(P)] = 1

    # Initialize V_list: for each row cluster p, assign each column to one of Qp column clusters
    V_list = []
    for p in range(P):
        Qp = Q_list[p]
        Vp = np.zeros((J, Qp), dtype=int)
        for j in range(J):
            Vp[j, rng.integers(Qp)] = 1
        V_list.append(Vp)

    return U, V_list


# Eq. 13
def update_C(X, U, V_list):
    """
    Update centroid matrices (C_blocks) for each row cluster.

    Parameters:
    - X: data matrix of shape (I, J)
    - U: binary matrix of shape (I, P) indicating row cluster membership
    - V_list: list of P matrices, each (J, Qp), indicating column cluster membership

    Returns:
    - C_blocks: list of centroid matrices, each of shape (1, Qp)
    """
    num_row_clusters = U.shape[1]
    C_blocks = []

    for p in range(num_row_clusters):
        up = U[:, p][:, None]  # (I, 1)
        Vp = V_list[p]  # (J, Qp)
        cluster_size = up.sum()
        Xp = up.T @ X  # (1, J)

        # Normalize Vp columns
        col_sums = Vp.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        Vp_norm = Vp / col_sums  # (J, Qp)

        if cluster_size > 0:
            centroid = Xp @ Vp_norm / cluster_size
        else:
            centroid = np.zeros((1, Vp.shape[1]))

        C_blocks.append(centroid)

    return C_blocks


# Eq. 14
def update_U(X, C_blocks, V_list, norm="l2"):
    """
    Update row-cluster assignment matrix U based on reconstruction error.

    Parameters:
    - X: data matrix of shape (I, J)
    - C_blocks: list of centroids, each of shape (1, Qp)
    - V_list: list of column cluster assignment matrices, each of shape (J, Qp)

    Returns:
    - new_U: binary matrix of shape (I, P), indicating row-cluster assignment
    """
    I, J = X.shape
    P = len(C_blocks)  # number of row clusters
    errors = np.zeros((I, P))  # error of assigning each row to each cluster

    for p in range(P):
        cp = C_blocks[p]  # centroid for cluster p (1, Qp)
        Vp = V_list[p]  # column assignment for cluster p (J, Qp)

        # Reconstruct entire data matrix using cluster p's centroid and column assignments
        reconstructed = cp @ Vp.T  # shape: (1, J)
        reconstructed = np.tile(reconstructed, (I, 1))  # broadcast to (I, J)

        # Compute reconstruction error for each row
        diff = X - reconstructed  # shape: (I, J)
        if norm == "l2":
            errors[:, p] = np.sum(diff**2, axis=1)  # shape: (I,)
        elif norm == "l1":
            errors[:, p] = np.sum(np.abs(diff), axis=1)  # shape: (I,)
        elif norm == "huber":
            errors[:, p] = np.sum(huber_loss(diff), axis=1)  # shape: (I,)
        elif norm == "mav_ratio":
            mae = np.sum(np.abs(diff), axis=1)  # mean absolute error per row
            mav = np.sum(np.abs(X), axis=1)  # mean absolute value per row
            errors[:, p] = mae / np.maximum(mav, 1e-12)  # MAV ratio per row
        else:
            raise ValueError(
                "Unsupported norm type: use 'l1', 'l2', 'huber', or 'mav_ratio'"
            )

    # Assign each row to the cluster with minimum reconstruction error
    new_U = np.zeros((I, P), dtype=int)
    assignments = np.argmin(errors, axis=1)  # best cluster index for each row
    new_U[np.arange(I), assignments] = 1  # binary encoding of cluster assignment

    # Ensure no cluster is left empty
    cluster_counts = new_U.sum(axis=0)
    empty_clusters = np.where(cluster_counts == 0)[0]

    for p in empty_clusters:
        # Find the row with the highest error for this cluster and assign it there
        worst_row = np.argmax(errors[:, p])
        new_U[worst_row] = 0
        new_U[worst_row, p] = 1

    return new_U


def update_V(X, U, C_blocks, Q_list, norm="l2"):
    """
    Update column-cluster assignment matrices V_list based on current row-cluster assignments and centroids.

    Parameters:
    - X: data matrix of shape (I, J)
    - U: binary row-cluster assignment matrix of shape (I, P)
    - C_blocks: list of centroids for each row cluster, each of shape (1, Qp)
    - Q_list: list of number of column clusters for each row cluster (length P)

    Returns:
    - V_list: list of updated binary column-cluster assignment matrices (each of shape (J, Qp))
    """
    J = X.shape[1]  # number of columns in data
    P = U.shape[1]  # number of row clusters
    V_list = []  # output: list of (J, Qp) matrices

    for p in range(P):
        Qp = Q_list[p]  # number of column clusters for row cluster p
        #  print(f"Updating V for row cluster {p} with {Qp} column clusters")
        Vp = np.zeros(
            (J, Qp), dtype=int
        )  # initialize column assignment matrix for cluster p

        # Get rows assigned to this row cluster
        cluster_rows = np.where(U[:, p] == 1)[0]

        # Skip if no rows assigned or no column clusters requested
        if len(cluster_rows) == 0 or Qp == 0:
            V_list.append(Vp)
            continue

        # Subset of X restricted to rows in row-cluster p
        Xp = X[cluster_rows]  # shape: (n_rows_p, J)
        cp = C_blocks[p]  # centroid block for cluster p, shape: (1, Qp)

        # Compute error for assigning each column to each of the Qp column clusters
        errors = np.empty((J, Qp))
        for j in range(J):
            if norm == "l2":
                # For column j, compute squared error with respect to each centroid in cp
                # Xp[:, j] is (n_rows_p,), cp is (1, Qp), broadcasting gives (n_rows_p, Qp)
                errors[j] = np.sum((Xp[:, j][:, None] - cp) ** 2, axis=0)
            elif norm == "l1":
                errors[j] = np.sum(np.abs(Xp[:, j][:, None] - cp), axis=0)
            elif norm == "huber":
                errors[j] = np.sum(huber_loss(Xp[:, j][:, None] - cp), axis=0)
            elif norm == "mav_ratio":
                diff = Xp[:, j][:, None] - cp  # (n_rows_p, Qp)
                mae = np.sum(np.abs(diff), axis=0)  # sum absolute error per centroid
                mav = np.sum(np.abs(Xp[:, j]))  # sum absolute value for column j
                errors[j] = mae / max(mav, 1e-12)  # MAV ratio per centroid
            else:
                raise ValueError(
                    "Unsupported norm type: use 'l1', 'l2', 'huber', or 'mav_ratio'"
                )

        # Assign each column j to the column cluster q with minimum reconstruction error
        assignments = np.argmin(errors, axis=1)
        for j, q in enumerate(assignments):
            Vp[j, q] = 1

        # Ensure all Qp column clusters are assigned at least one column
        counts = Vp.sum(axis=0)
        missing = np.where(counts == 0)[0]

        for q in missing:
            # Assign to cluster q the column with the highest error for q
            j = np.argmax(errors[:, q])
            Vp[j] = 0
            Vp[j, q] = 1

        V_list.append(Vp)

    return V_list


def compute_loss(X, U, C_blocks, V_list, norm="l2"):
    """
    Compute the reconstruction loss for the current clustering configuration.

    Parameters:
    - X: data matrix of shape (I, J)
    - U: binary matrix of shape (I, P), row-cluster assignments
    - C_blocks: list of centroid blocks, each of shape (1, Qp)
    - V_list: list of binary column-cluster matrices (J, Qp)

    Returns:
    - loss: total squared reconstruction error
    """
    P = U.shape[1]  # number of row clusters
    loss = 0

    for p in range(P):
        up = U[:, p][:, None]  # (I, 1): mask for rows in cluster p
        Vp = V_list[p]  # (J, Qp): column-cluster assignment matrix
        cp = C_blocks[p]  # (1, Qp): cluster p’s column centroids

        # Reconstruct the block for cluster p: up @ (cp @ Vp.T) → shape (I, J)
        reconstruction = up @ (cp @ Vp.T)

        # Only compute error for rows assigned to cluster p
        mask = up == 1
        diff = (X - reconstruction) * mask  # zero out rows not in cluster p

        if norm == "l2":
            loss += np.sum(diff**2)
        elif norm == "l1":
            loss += np.sum(np.abs(diff))
        elif norm == "huber":
            loss += np.sum(huber_loss(diff))
        elif norm == "mav_ratio":
            mae = np.sum(np.abs(diff))
            mav = np.sum(np.abs(X * mask))  # only consider rows in cluster p
            loss += mae / max(mav, 1e-12)  # MAV ratio for cluster p
        else:
            raise ValueError(
                "Unsupported norm type: use 'l1', 'l2', 'huber', or 'mav_ratio'"
            )

    return loss


def generalized_double_kmeans(
    X, P, Q_list, max_iter=100, tol=1e-4, random_state=None, norm="l2"
):
    I, J = X.shape
    U, V_list = initialize_partitions(I, J, P, Q_list, random_state=random_state)
    prev_loss = np.inf
    for i in range(max_iter):
        C_blocks = update_C(X, U, V_list)
        for idx, cp in enumerate(C_blocks):
            if np.any(np.isnan(cp)):
                print(f"NaNs in C_blocks[{idx}]")
        U = update_U(X, C_blocks, V_list, norm=norm)
        V_list = update_V(X, U, C_blocks, Q_list, norm=norm)
        loss = compute_loss(X, U, C_blocks, V_list, norm=norm)
        print(
            f"Iteration {i}: Loss={loss:.2e}, max X={X.max()}, max cp={max(np.max(c) for c in C_blocks)}"
        )
        if abs(prev_loss - loss) < tol:
            print(f"Converged at iteration {i}")
            break
        prev_loss = loss
    return U, V_list, C_blocks, loss


def compute_gdkm_cv_scores(
    data, P_range=range(2, 6), Q_range=range(2, 6), cv_folds=3, true_row_labels=None
):

    X = StandardScaler().fit_transform(data)
    results = []

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for P in P_range:
        for Q in Q_range:
            losses, silhouettes, aris = [], [], []
            Q_list = [Q] * P

            for train_index, test_index in kf.split(X):
                X_train = X[train_index, :]
                X_test = X[test_index, :]

                try:
                    U, V, C_blocks, _ = generalized_double_kmeans(
                        X_train, P=P, Q_list=Q_list, random_state=42
                    )
                    # print(
                    #     f"P={P}, row cluster counts: {np.bincount(np.argmax(U, axis=1))}"
                    # )

                    # Reconstruct V_list from V and print column cluster counts per block
                    col_start = 0
                    V_list = []
                    for p in range(P):
                        Qp = Q_list[p]
                        Vp = V[:, col_start : col_start + Qp]
                        cluster_counts = np.bincount(
                            np.argmax(Vp, axis=1), minlength=Qp
                        )
                        print(
                            f"P={P}, Q_p={Qp}, Col cluster counts for block {p}: {cluster_counts}"
                        )
                        V_list.append(Vp)
                        col_start += Qp

                    # Assign clusters to test set based on minimal reconstruction error
                    test_labels = np.zeros((X_test.shape[0], P))
                    for i, xi in enumerate(X_test):
                        errors = []
                        col_start = 0
                        for p in range(P):
                            Qp = Q_list[p]
                            Vp = V[:, col_start : col_start + Qp]
                            cp = C_blocks[p]
                            reconstruction = cp @ Vp.T
                            err = np.linalg.norm(xi - reconstruction, ord=2)
                            errors.append(err)
                            col_start += Qp
                        best_p = np.argmin(errors)
                        test_labels[i, best_p] = 1

                    loss = compute_loss(X_test, test_labels, C_blocks, V_list)
                    losses.append(loss)

                    row_labels = np.argmax(test_labels, axis=1)
                    ari = (
                        adjusted_rand_score(true_row_labels[test_index], row_labels)
                        if true_row_labels is not None
                        else np.nan
                    )

                    try:
                        sil = silhouette_score(X_test, row_labels)
                    except:
                        sil = np.nan

                    silhouettes.append(sil)
                    aris.append(ari)

                except Exception:
                    losses.append(np.nan)
                    silhouettes.append(np.nan)
                    aris.append(np.nan)

            results.append(
                {
                    "P": P,
                    "Q": Q,
                    "Mean Loss": safe_mean(losses),
                    "Mean Silhouette": safe_mean(silhouettes),
                    "Mean ARI": safe_mean(aris),
                }
            )

    return pd.DataFrame(results)


def suggest_optimal_pq(results_df, criterion="silhouette", penalty_lambda=0.0):
    """
    Suggests the best (P, Q) pair based on the chosen criterion.

    Parameters:
    - results_df: DataFrame returned by compute_gdkm_cv_scores
    - criterion: 'silhouette', 'loss', or 'bic'
    - penalty_lambda: regularization weight for BIC-like loss (only used if criterion == 'bic')

    Returns:
    - best_pq: (P, Q) tuple with best score
    - best_score: the score associated with that (P, Q)
    """
    df = results_df.copy()

    if criterion == "silhouette":
        idx = df["Mean Silhouette"].idxmax()
        best_score = df.loc[idx, "Mean Silhouette"]
    elif criterion == "loss":
        idx = df["Mean Loss"].idxmin()
        best_score = df.loc[idx, "Mean Loss"]
    elif criterion == "bic":
        # Penalized score = loss + lambda * (P + Q)
        df["Penalized Loss"] = df["Mean Loss"] + penalty_lambda * (df["P"] + df["Q"])
        idx = df["Penalized Loss"].idxmin()
        best_score = df.loc[idx, "Penalized Loss"]
    else:
        raise ValueError("criterion must be 'silhouette', 'loss', or 'bic'")

    best_p = df.loc[idx, "P"]
    best_q = df.loc[idx, "Q"]
    return (int(best_p), int(best_q)), best_score


def estimate_pq_with_umap_hdbscan(
    X, min_cluster_size=5, n_neighbors=15, min_dist=0.1, random_state=42, scale=True
):
    """
    Estimate the number of row (P) and column (Q) clusters using UMAP + HDBSCAN.

    Parameters:
        X : ndarray of shape (I, J)
            Input data matrix.
        min_cluster_size : int
            Minimum cluster size for HDBSCAN.
        n_neighbors : int
            UMAP neighborhood size.
        min_dist : float
            Minimum UMAP distance parameter.
        random_state : int
            Seed for reproducibility.

    Returns:
        P_est : int
            Estimated number of row clusters.
        Q_est : int
            Estimated number of column clusters.
    """
    I, J = X.shape

    # Normalize rows and columns
    if scale:
        row_scaled = StandardScaler().fit_transform(X)
        col_scaled = StandardScaler().fit_transform(X.T)
    else:
        row_scaled = X
        col_scaled = X.T

    # UMAP embedding
    row_embed = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state
    ).fit_transform(row_scaled)
    col_embed = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state
    ).fit_transform(col_scaled)

    # HDBSCAN clustering
    row_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(row_embed)
    col_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(col_embed)

    # Count valid (non-noise) clusters
    P_est = len(set(row_clusterer.labels_)) - (1 if -1 in row_clusterer.labels_ else 0)
    Q_est = len(set(col_clusterer.labels_)) - (1 if -1 in col_clusterer.labels_ else 0)

    return P_est, Q_est
