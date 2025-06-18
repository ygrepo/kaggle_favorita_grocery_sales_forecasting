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
    ):
        self.n_row_clusters = n_row_clusters
        self.n_col_clusters_list = n_col_clusters_list
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        self.U_, self.V_list_, self.C_blocks_, self.loss_ = generalized_double_kmeans(
            X,
            P=self.n_row_clusters,
            Q_list=self.n_col_clusters_list,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
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


def initialize_partitions(I, J, P, Q_list, random_state=None):
    rng = np.random.default_rng(random_state)
    # U: (I, P)
    U = np.zeros((I, P), dtype=int)
    for i in range(I):
        U[i, rng.integers(P)] = 1
    V_list = []
    for p in range(P):
        Qp = Q_list[p]
        # Vp: (J, Qp)
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
def update_U(X, C_blocks, V_list):
    I, J = X.shape
    P = len(C_blocks)
    errors = np.zeros((I, P))

    for p in range(P):
        cp = C_blocks[p]  # (1, Qp)
        Vp = V_list[p]  # (J, Qp)
        reconstruction = cp @ Vp.T  # (1, J)
        reconstruction = np.tile(reconstruction, (I, 1))  # (I, J)
        diff = X - reconstruction
        errors[:, p] = np.sum(diff**2, axis=1)

    # Assign each row to cluster with min error
    new_U = np.zeros((I, P), dtype=int)
    assignments = np.argmin(errors, axis=1)
    new_U[np.arange(I), assignments] = 1

    # Ensure all row clusters are non-empty
    counts = new_U.sum(axis=0)
    missing = np.where(counts == 0)[0]

    for p in missing:
        # Reassign row with highest current error to this empty cluster
        worst_row = np.argmax(errors[:, p])
        new_U[worst_row] = 0
        new_U[worst_row, p] = 1

    return new_U


def update_V(X, U, C_blocks, Q_list):
    J = X.shape[1]
    P = U.shape[1]
    V_list = []

    for p in range(P):
        Qp = Q_list[p]
        Vp = np.zeros((J, Qp), dtype=int)
        cluster_rows = np.where(U[:, p] == 1)[0]

        if len(cluster_rows) == 0 or Qp == 0:
            V_list.append(Vp)
            continue

        Xp = X[cluster_rows]  # (n_rows_p, J)
        cp = C_blocks[p]  # (1, Qp)

        errors = np.empty((J, Qp))
        for j in range(J):
            # Correct shape: (n_rows_p, 1) - (1, Qp) → (n_rows_p, Qp)
            errors[j] = np.sum((Xp[:, j][:, None] - cp) ** 2, axis=0)

        assignments = np.argmin(errors, axis=1)
        for j, q in enumerate(assignments):
            Vp[j, q] = 1

        # Ensure all Qp clusters are represented
        counts = Vp.sum(axis=0)
        missing = np.where(counts == 0)[0]
        for q in missing:
            j = np.argmax(errors[:, q])  # column with max error for q
            Vp[j] = 0
            Vp[j, q] = 1

        V_list.append(Vp)

    return V_list


def compute_loss(X, U, C_blocks, V_list):
    P = U.shape[1]  # number of row clusters
    loss = 0

    for p in range(P):
        up = U[:, p][:, None]  # (I, 1)
        Vp = V_list[p]  # (J, Qp)
        cp = C_blocks[p]  # (1, Qp)

        # Compute reconstruction for assigned rows
        reconstruction = up @ (cp @ Vp.T)  # (I, J)

        # Only keep rows assigned to this cluster (up == 1), else zero out
        mask = up == 1
        diff = (X - reconstruction) * mask  # (I, J)

        loss += np.sum(diff**2)

    return loss


def generalized_double_kmeans(X, P, Q_list, max_iter=100, tol=1e-4, random_state=None):
    I, J = X.shape
    U, V_list = initialize_partitions(I, J, P, Q_list, random_state=random_state)
    prev_loss = np.inf
    for i in range(max_iter):
        C_blocks = update_C(X, U, V_list)
        for idx, cp in enumerate(C_blocks):
            if np.any(np.isnan(cp)):
                print(f"NaNs in C_blocks[{idx}]")
        U = update_U(X, C_blocks, V_list)
        V_list = update_V(X, U, C_blocks, Q_list)
        loss = compute_loss(X, U, C_blocks, V_list)
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
