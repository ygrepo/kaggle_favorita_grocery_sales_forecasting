# Generalized Double K-Means (GDKM) implementation
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.base import BaseEstimator, BiclusterMixin


class GeneralizedDoubleKMeans(BaseEstimator, BiclusterMixin):
    def __init__(
        self,
        n_row_clusters=3,
        n_col_clusters_list=None,
        max_iter=100,
        tol=1e-4,
        random_state=None,
        norm="l2",
    ):
        self.n_row_clusters = n_row_clusters
        self.n_col_clusters_list = n_col_clusters_list
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.norm = norm

    def fit(self, X):
        self.U_, self.V_list_, self.C_blocks_, self.loss_ = generalized_double_kmeans(
            X,
            P=self.n_row_clusters,
            Q_list=self.n_col_clusters_list,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            norm=self.norm,
        )

        self.row_labels_ = np.argmax(self.U_, axis=1)

        # Ensure all Vp are 2D before concatenation
        V_nonempty = [Vp for Vp in self.V_list_ if Vp.ndim == 2 and Vp.shape[1] > 0]
        if len(V_nonempty) == 0:
            raise ValueError("All column cluster blocks are empty or malformed.")

        V_concat = np.concatenate(V_nonempty, axis=1)
        self.column_labels_ = np.argmax(V_concat, axis=1)

        return self

    def get_row_clusters(self):
        return self.row_labels_

    def get_column_clusters(self):
        return self.column_labels_

    def get_biclusters(self):
        """
        Return boolean indicator matrices (rows, columns) for each (p, q) bicluster.
        Each bicluster corresponds to a pair of (row cluster p, column cluster q).

        Returns:
            rows: (n_biclusters, n_rows) boolean array
            cols: (n_biclusters, n_columns) boolean array
        """
        if not hasattr(self, "U_") or not hasattr(self, "V_list_"):
            raise AttributeError("Must call fit() before get_biclusters().")

        row_cluster_count = self.U_.shape[1]
        rows = []
        cols = []

        for p in range(row_cluster_count):
            row_mask = self.U_[:, p] == 1  # (n_rows,)
            Vp = self.V_list_[p]  # (n_cols, Qp)
            Qp = Vp.shape[1]

            for q in range(Qp):
                col_mask = Vp[:, q] == 1  # (n_cols,)
                if np.any(row_mask) and np.any(col_mask):  # optional safeguard
                    rows.append(row_mask)
                    cols.append(col_mask)

        return np.array(rows), np.array(cols)

    @property
    def biclusters_(self):
        """
        Alias property to comply with sklearn's BiclusterMixin standard.
        Matches SpectralBiclustering.biclusters_ structure.
        """
        return self.get_biclusters()


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

        # Compute squared reconstruction error for each row
        diff = X - reconstructed  # shape: (I, J)
        if norm == "l2":
            errors[:, p] = np.sum(diff**2, axis=1)  # shape: (I,)
        elif norm == "l1":
            errors[:, p] = np.sum(np.abs(diff), axis=1)  # shape: (I,)
        elif norm == "huber":
            errors[:, p] = np.sum(huber_loss(diff), axis=1)  # shape: (I,)
        else:
            raise ValueError("Unsupported norm type: use 'l1' or 'l2'")

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
            else:
                raise ValueError("Unsupported norm type: use 'l1' or 'l2'")

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
        else:
            raise ValueError("Unsupported norm type: use 'l1' or 'l2'")

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
