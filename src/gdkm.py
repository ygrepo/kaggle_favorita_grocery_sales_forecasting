# Generalized Double K-Means (GDKM) implementation
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import pandas as pd


def _safe_mean(arr):
    return np.nan if np.all(np.isnan(arr)) else np.nanmean(arr)


def initialize_partitions(I, J, P, Q_list, random_state=None):
    rng = np.random.default_rng(random_state)
    U = np.zeros((I, P), dtype=int)
    for i in range(I):
        U[i, rng.integers(P)] = 1
    V_list = []
    for p in range(P):
        Qp = Q_list[p]
        Vp = np.zeros((J, Qp), dtype=int)
        for j in range(J):
            Vp[j, rng.integers(Qp)] = 1
        V_list.append(Vp)
    return U, V_list


# Eq. 13
def update_C_fixed(X, U, V_list):
    P = U.shape[1]
    C_blocks = []
    for p in range(P):
        up = U[:, p][:, None]  # shape: (I, 1)
        Vp = V_list[p]  # shape: (J, Qp)
        Xp = up.T @ X  # shape: (1, J)
        cluster_size = up.sum()
        cp = Xp @ Vp / cluster_size if cluster_size > 0 else np.zeros((1, Vp.shape[1]))
        C_blocks.append(cp)
    return C_blocks


# def update_C_soft(X, U, V_list):
#     """
#     Update centroids C when V_p are real-valued (soft cluster assignments).

#     Parameters:
#     - X: data matrix (I x J)
#     - U: row assignment matrix (I x P)
#     - V_list: list of P variable cluster assignment matrices (J x Q_p), real-valued

#     Returns:
#     - C_blocks: list of P centroid matrices, each of shape (1 x Q_p)
#     """
#     P = U.shape[1]
#     C_blocks = []

#     for p in range(P):
#         up = U[:, p][:, None]  # (I, 1)
#         Vp = V_list[p]  # (J, Q_p)

#         cluster_size = up.sum()
#         if cluster_size > 0:
#             # (1 x J) @ (J x Qp) = (1 x Qp)
#             Xp = up.T @ X  # (1 x J)
#             VpT_Vp_inv = np.linalg.pinv(Vp.T @ Vp)  # (Qp x Qp)
#             cp = (Xp @ Vp) @ VpT_Vp_inv / cluster_size
#         else:
#             cp = np.zeros((1, Vp.shape[1]))

#         C_blocks.append(cp)

#     return C_blocks


# Eq. 14
def update_U(X, C_blocks, V_list):
    I, J = X.shape
    P = len(C_blocks)
    errors = np.zeros((I, P))
    for p in range(P):
        cp = C_blocks[p]
        Vp = V_list[p]
        reconstruction = cp @ Vp.T
        reconstruction = np.tile(reconstruction, (I, 1))
        diff = X - reconstruction
        errors[:, p] = np.sum(diff**2, axis=1)
    new_U = np.zeros((I, P), dtype=int)
    new_U[np.arange(I), np.argmin(errors, axis=1)] = 1
    return new_U


# def update_U_soft(X, C_blocks, V_list, m=2.0, eps=1e-8):
#     """
#     Soft update of U (fuzzy membership matrix) based on reconstruction errors.

#     Parameters:
#     - X: data matrix (I x J)
#     - C_blocks: list of centroids (each (1 x Qp))
#     - V_list: list of variable clusters (each (J x Qp))
#     - m: fuzzifier (float > 1)
#     - eps: small value to avoid division by zero

#     Returns:
#     - U_soft: soft membership matrix (I x P), rows sum to 1
#     """
#     I, J = X.shape
#     P = len(C_blocks)
#     distances = np.zeros((I, P))

#     for p in range(P):
#         cp = C_blocks[p]  # (1 x Qp)
#         Vp = V_list[p]  # (J x Qp)
#         recon = cp @ Vp.T  # (1 x J)
#         diff = X - recon  # broadcasted (I x J)
#         distances[:, p] = np.sqrt(np.sum(diff**2, axis=1)) + eps

#     # Fuzzy memberships: inverse-distance weighted
#     inv_dist = distances ** (-2 / (m - 1))
#     U_soft = inv_dist / inv_dist.sum(axis=1, keepdims=True)

#     return U_soft


def update_V(X, U, C_blocks, Q_list):
    J = X.shape[1]
    P = U.shape[1]
    V_list = []
    for p in range(P):
        Qp = Q_list[p]
        Vp = np.zeros((J, Qp), dtype=int)
        cluster_rows = np.where(U[:, p] == 1)[0]
        if len(cluster_rows) == 0:
            V_list.append(Vp)
            continue
        Xp = X[cluster_rows]
        cp = C_blocks[p]
        for j in range(J):
            errors = np.sum((Xp[:, j][:, None] - cp[:, :]) ** 2, axis=0)
            q = np.argmin(errors)
            Vp[j, q] = 1
        V_list.append(Vp)
    return V_list


def compute_loss(X, U, C_blocks, V_list):
    P = U.shape[1]
    loss = 0
    for p in range(P):
        up = U[:, p][:, None]
        Vp = V_list[p]
        cp = C_blocks[p]
        reconstruction = up @ (cp @ Vp.T)
        diff = up * X - reconstruction
        loss += np.sum(diff**2)
    return loss


def generalized_double_kmeans(X, P, Q_list, max_iter=100, tol=1e-4, random_state=None):
    I, J = X.shape
    U, V_list = initialize_partitions(I, J, P, Q_list, random_state=random_state)
    prev_loss = np.inf
    for _ in range(max_iter):
        C_blocks = update_C_fixed(X, U, V_list)
        U = update_U(X, C_blocks, V_list)
        V_list = update_V(X, U, C_blocks, Q_list)
        loss = compute_loss(X, U, C_blocks, V_list)
        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
    V = np.concatenate(V_list, axis=1)
    return U, V, C_blocks, loss


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
