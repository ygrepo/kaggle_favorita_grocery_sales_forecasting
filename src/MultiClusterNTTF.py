import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
import warnings


class MultiClusterNTTF(BaseEstimator, TransformerMixin):
    """
    Multi-Cluster Non-negative Tensor Tri-Factorization

    Provides explicit multiple cluster memberships for rows, columns, and features
    in a 3D tensor X of shape (I, J, K).

    Parameters
    ----------
    R1 : int
        Number of row clusters
    R2 : int
        Number of column clusters
    R3 : int
        Number of feature clusters
    sparsity_penalty : float, default=0.0
        L1 penalty to encourage sparse (fewer) cluster memberships
    membership_threshold : float, default=0.1
        Threshold for determining cluster membership (relative to max)
    max_clusters_per_entity : int, default=None
        Maximum clusters an entity can belong to (None = no limit)
    algorithm : str, default='Frobenius'
        Algorithm to use ('Frobenius', 'KL', 'IS')
    init : str, default='Random'
        Initialization method ('Random', 'SVD', 'KMeans')
    thr : float, default=1e-10
        Convergence threshold
    num_iter : int, default=100
        Maximum number of iterations
    verbose : bool, default=False
        Whether to print verbose output
    """

    def __init__(
        self,
        R1,
        R2,
        R3,
        sparsity_penalty=0.0,
        membership_threshold=0.1,
        max_clusters_per_entity=None,
        algorithm="Frobenius",
        init="Random",
        thr=1e-10,
        num_iter=100,
        verbose=False,
    ):
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.sparsity_penalty = sparsity_penalty
        self.membership_threshold = membership_threshold
        self.max_clusters_per_entity = max_clusters_per_entity
        self.algorithm = algorithm
        self.init = init
        self.thr = thr
        self.num_iter = num_iter
        self.verbose = verbose

        # Factors
        self.A = None  # Row factor (I, R1)
        self.B = None  # Column factor (J, R2)
        self.C = None  # Feature factor (K, R3)
        self.S = None  # Core tensor (R1, R2, R3)

        # Cluster memberships
        self.row_clusters_ = None
        self.col_clusters_ = None
        self.feature_clusters_ = None

        # Training history
        self.RecError = []
        self.RelChange = []
        self.n_iter_ = 0

    def _check_input(self, X):
        """Validate input tensor"""
        if len(X.shape) != 3:
            raise ValueError("Input must be a 3D tensor with shape (I, J, K)")
        if np.any(X < 0):
            raise ValueError("Input tensor must be non-negative")
        return X

    def _tensor_frobenius_norm(self, tensor):
        """Compute Frobenius norm of a tensor (works for any dimension)"""
        return np.sqrt(np.sum(tensor**2))

    def _initialize_factors(self, X):
        """Initialize factors with cluster-friendly initialization"""
        I, J, K = X.shape

        if self.init == "KMeans":
            try:
                # Initialize using K-means for better cluster structure
                from sklearn.cluster import KMeans

                # Row clusters from mode-1 unfolding
                X1 = X.reshape(I, -1)
                kmeans1 = KMeans(
                    n_clusters=self.R1, random_state=42, n_init=10
                )
                row_labels = kmeans1.fit_predict(X1)
                self.A = np.zeros((I, self.R1))
                for i, label in enumerate(row_labels):
                    self.A[i, label] = 1.0
                # Add small random noise
                self.A += 0.1 * np.random.rand(I, self.R1)

                # Column clusters from mode-2 unfolding
                X2 = X.transpose(1, 0, 2).reshape(J, -1)
                kmeans2 = KMeans(
                    n_clusters=self.R2, random_state=42, n_init=10
                )
                col_labels = kmeans2.fit_predict(X2)
                self.B = np.zeros((J, self.R2))
                for j, label in enumerate(col_labels):
                    self.B[j, label] = 1.0
                self.B += 0.1 * np.random.rand(J, self.R2)

                # Feature clusters from mode-3 unfolding
                X3 = X.transpose(2, 0, 1).reshape(K, -1)
                kmeans3 = KMeans(
                    n_clusters=self.R3, random_state=42, n_init=10
                )
                feat_labels = kmeans3.fit_predict(X3)
                self.C = np.zeros((K, self.R3))
                for k, label in enumerate(feat_labels):
                    self.C[k, label] = 1.0
                self.C += 0.1 * np.random.rand(K, self.R3)

            except ImportError:
                if self.verbose:
                    print("sklearn not available, using random initialization")
                self.A = np.random.rand(I, self.R1)
                self.B = np.random.rand(J, self.R2)
                self.C = np.random.rand(K, self.R3)

        elif self.init == "SVD":
            # SVD initialization
            X1 = X.reshape(I, -1)
            U1, _, _ = np.linalg.svd(X1, full_matrices=False)
            self.A = np.abs(U1[:, : self.R1]) + 1e-6

            X2 = X.transpose(1, 0, 2).reshape(J, -1)
            U2, _, _ = np.linalg.svd(X2, full_matrices=False)
            self.B = np.abs(U2[:, : self.R2]) + 1e-6

            X3 = X.transpose(2, 0, 1).reshape(K, -1)
            U3, _, _ = np.linalg.svd(X3, full_matrices=False)
            self.C = np.abs(U3[:, : self.R3]) + 1e-6

        else:  # Random
            self.A = np.random.rand(I, self.R1)
            self.B = np.random.rand(J, self.R2)
            self.C = np.random.rand(K, self.R3)

        # Initialize core tensor
        self.S = np.random.rand(self.R1, self.R2, self.R3)

        # Ensure non-negativity
        self.A = np.maximum(self.A, 1e-16)
        self.B = np.maximum(self.B, 1e-16)
        self.C = np.maximum(self.C, 1e-16)
        self.S = np.maximum(self.S, 1e-16)

    def _reconstruct_tensor(self):
        """Reconstruct tensor from factors"""
        I, J, K = self.A.shape[0], self.B.shape[0], self.C.shape[0]
        X_rec = np.zeros((I, J, K))

        for k in range(K):
            for r3 in range(self.R3):
                X_rec[:, :, k] += self.C[k, r3] * (
                    self.A @ self.S[:, :, r3] @ self.B.T
                )

        return X_rec

    def _apply_sparsity_penalty(self, factor):
        """Apply L1 sparsity penalty but protect at least one cluster per row"""
        if self.sparsity_penalty > 0:
            factor_sparse = np.copy(factor)

            for i in range(factor.shape[0]):
                row = factor[i, :]

                # Soft thresholding
                row_sparse = np.sign(row) * np.maximum(
                    np.abs(row) - self.sparsity_penalty, 0
                )

                # If everything got zeroed, keep strongest
                if np.sum(np.abs(row_sparse)) < 1e-10:
                    max_idx = np.argmax(np.abs(row))
                    row_sparse[max_idx] = row[max_idx]

                factor_sparse[i, :] = row_sparse

            return np.maximum(factor_sparse, 1e-16)
        else:
            # No sparsity penalty
            return np.maximum(factor, 1e-16)

    def _enforce_max_clusters(self, factor):
        """Enforce maximum number of clusters per entity"""
        if self.max_clusters_per_entity is not None:
            for i in range(factor.shape[0]):
                # Keep only top-k cluster memberships
                if factor.shape[1] > self.max_clusters_per_entity:
                    top_k_indices = np.argsort(factor[i])[
                        -self.max_clusters_per_entity :
                    ]
                    mask = np.zeros_like(factor[i], dtype=bool)
                    mask[top_k_indices] = True
                    factor[i, ~mask] = 1e-16
        return factor

    def _update_frobenius_with_constraints(self, X):
        """Update with sparsity and cluster constraints"""
        I, J, K = X.shape
        X_rec = self._reconstruct_tensor()

        # Update A with constraints
        numerator_A = np.zeros((I, self.R1))
        denominator_A = np.zeros((I, self.R1))

        for k in range(K):
            for r3 in range(self.R3):
                weight = self.C[k, r3]
                S_slice = self.S[:, :, r3]  # (R1, R2)

                numerator_A += weight * X[:, :, k] @ self.B @ S_slice.T
                denominator_A += weight * X_rec[:, :, k] @ self.B @ S_slice.T

        self.A *= numerator_A / (denominator_A + 1e-16)
        self.A = self._apply_sparsity_penalty(self.A)
        self.A = self._enforce_max_clusters(self.A)

        # Update B with constraints
        X_rec = self._reconstruct_tensor()
        numerator_B = np.zeros((J, self.R2))
        denominator_B = np.zeros((J, self.R2))

        for k in range(K):
            for r3 in range(self.R3):
                weight = self.C[k, r3]
                S_slice = self.S[:, :, r3]  # (R1, R2)

                numerator_B += weight * X[:, :, k].T @ self.A @ S_slice
                denominator_B += weight * X_rec[:, :, k].T @ self.A @ S_slice

        self.B *= numerator_B / (denominator_B + 1e-16)
        self.B = self._apply_sparsity_penalty(self.B)
        self.B = self._enforce_max_clusters(self.B)

        # Update C with constraints
        X_rec = self._reconstruct_tensor()
        numerator_C = np.zeros((K, self.R3))
        denominator_C = np.zeros((K, self.R3))

        for k in range(K):
            for r3 in range(self.R3):
                S_slice = self.S[:, :, r3]
                ASB = self.A @ S_slice @ self.B.T

                numerator_C[k, r3] = np.sum(X[:, :, k] * ASB)
                denominator_C[k, r3] = np.sum(X_rec[:, :, k] * ASB)

        self.C *= numerator_C / (denominator_C + 1e-16)
        self.C = self._apply_sparsity_penalty(self.C)
        self.C = self._enforce_max_clusters(self.C)

        # Update S (core tensor)
        X_rec = self._reconstruct_tensor()
        numerator_S = np.zeros((self.R1, self.R2, self.R3))
        denominator_S = np.zeros((self.R1, self.R2, self.R3))

        for r3 in range(self.R3):
            # Compute weighted sum over features
            X_weighted = np.zeros((I, J))
            X_rec_weighted = np.zeros((I, J))

            for k in range(K):
                weight = self.C[k, r3]
                X_weighted += weight * X[:, :, k]
                X_rec_weighted += weight * X_rec[:, :, k]

            numerator_S[:, :, r3] = self.A.T @ X_weighted @ self.B
            denominator_S[:, :, r3] = self.A.T @ X_rec_weighted @ self.B

        self.S *= numerator_S / (denominator_S + 1e-16)
        self.S = np.maximum(self.S, 1e-16)

    def _extract_cluster_memberships(self):
        """Extract explicit cluster memberships from factors"""
        # Normalize factors to get membership probabilities
        A_norm = self.A / (self.A.sum(axis=1, keepdims=True) + 1e-16)
        B_norm = self.B / (self.B.sum(axis=1, keepdims=True) + 1e-16)
        C_norm = self.C / (self.C.sum(axis=1, keepdims=True) + 1e-16)

        # Extract memberships based on threshold
        self.row_clusters_ = []
        for i in range(A_norm.shape[0]):
            max_val = A_norm[i].max()
            threshold = max_val * self.membership_threshold
            clusters = np.where(A_norm[i] >= threshold)[0].tolist()
            self.row_clusters_.append(clusters)

        self.col_clusters_ = []
        for j in range(B_norm.shape[0]):
            max_val = B_norm[j].max()
            threshold = max_val * self.membership_threshold
            clusters = np.where(B_norm[j] >= threshold)[0].tolist()
            self.col_clusters_.append(clusters)

        self.feature_clusters_ = []
        for k in range(C_norm.shape[0]):
            max_val = C_norm[k].max()
            threshold = max_val * self.membership_threshold
            clusters = np.where(C_norm[k] >= threshold)[0].tolist()
            self.feature_clusters_.append(clusters)

    def fit(self, X, y=None):
        """Fit the model"""
        X = self._check_input(X)
        I, J, K = X.shape

        if self.verbose:
            print(f"Fitting MultiClusterNTTF on tensor {X.shape}")
            print(
                f"Clusters: {self.R1} rows, {self.R2} cols, {self.R3} features"
            )

        self._initialize_factors(X)
        self.RecError = []
        self.RelChange = []

        for iteration in range(self.num_iter):
            # Store previous factors
            A_prev = self.A.copy()
            B_prev = self.B.copy()
            C_prev = self.C.copy()
            S_prev = self.S.copy()

            # Update with constraints
            self._update_frobenius_with_constraints(X)

            # Compute metrics
            X_rec = self._reconstruct_tensor()
            rec_error = self._tensor_frobenius_norm(X - X_rec) ** 2
            self.RecError.append(rec_error)

            # Relative change - FIXED: Use proper tensor norm
            rel_changes = [
                np.linalg.norm(self.A - A_prev, "fro")
                / (np.linalg.norm(A_prev, "fro") + 1e-16),
                np.linalg.norm(self.B - B_prev, "fro")
                / (np.linalg.norm(B_prev, "fro") + 1e-16),
                np.linalg.norm(self.C - C_prev, "fro")
                / (np.linalg.norm(C_prev, "fro") + 1e-16),
                self._tensor_frobenius_norm(self.S - S_prev)
                / (self._tensor_frobenius_norm(S_prev) + 1e-16),
            ]
            rel_change = max(rel_changes)
            self.RelChange.append(rel_change)

            if self.verbose and (iteration + 1) % 20 == 0:
                print(
                    f"Iter {iteration + 1}: Error = {rec_error:.6f}, RelChange = {rel_change:.6f}"
                )

            if rel_change < self.thr:
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break

        self.n_iter_ = iteration + 1

        # Extract cluster memberships
        self._extract_cluster_memberships()

        return self

    def get_cluster_memberships(self, mode="all"):
        """
        Get cluster memberships for entities

        Parameters
        ----------
        mode : str, default='all'
            Which memberships to return ('rows', 'cols', 'features', 'all')

        Returns
        -------
        memberships : dict or list
            Cluster memberships for each entity
        """
        if self.row_clusters_ is None:
            raise ValueError("Model must be fitted first")

        if mode == "rows":
            return self.row_clusters_
        elif mode == "cols":
            return self.col_clusters_
        elif mode == "features":
            return self.feature_clusters_
        elif mode == "all":
            return {
                "rows": self.row_clusters_,
                "cols": self.col_clusters_,
                "features": self.feature_clusters_,
            }
        else:
            raise ValueError(
                "mode must be 'rows', 'cols', 'features', or 'all'"
            )

    def get_membership_strengths(self, mode="all"):
        """
        Get membership strength matrices (soft memberships)

        Returns
        -------
        strengths : dict
            Normalized membership strength matrices
        """
        if self.A is None:
            raise ValueError("Model must be fitted first")

        A_norm = self.A / (self.A.sum(axis=1, keepdims=True) + 1e-16)
        B_norm = self.B / (self.B.sum(axis=1, keepdims=True) + 1e-16)
        C_norm = self.C / (self.C.sum(axis=1, keepdims=True) + 1e-16)

        if mode == "rows":
            return A_norm
        elif mode == "cols":
            return B_norm
        elif mode == "features":
            return C_norm
        elif mode == "all":
            return {"rows": A_norm, "cols": B_norm, "features": C_norm}

    def get_factors(self):
        """Get all factor matrices and core tensor"""
        return {"A": self.A, "B": self.B, "C": self.C, "S": self.S}

    def inverse_transform(self):
        """Reconstruct the original tensor"""
        if self.A is None:
            raise ValueError("Model must be fitted first")
        return self._reconstruct_tensor()

    def print_cluster_summary(self):
        """Print summary of cluster memberships"""
        if self.row_clusters_ is None:
            raise ValueError("Model must be fitted first")

        print("=== Cluster Membership Summary ===")

        # Row clusters
        row_counts = [len(clusters) for clusters in self.row_clusters_]
        print(f"\nRows ({len(self.row_clusters_)} total):")
        print(
            f"  Multiple memberships: {sum(1 for c in row_counts if c > 1)} rows"
        )
        print(f"  Avg clusters per row: {np.mean(row_counts):.2f}")
        print(f"  Max clusters per row: {max(row_counts)}")

        # Column clusters
        col_counts = [len(clusters) for clusters in self.col_clusters_]
        print(f"\nColumns ({len(self.col_clusters_)} total):")
        print(
            f"  Multiple memberships: {sum(1 for c in col_counts if c > 1)} columns"
        )
        print(f"  Avg clusters per column: {np.mean(col_counts):.2f}")
        print(f"  Max clusters per column: {max(col_counts)}")

        # Feature clusters
        feat_counts = [len(clusters) for clusters in self.feature_clusters_]
        print(f"\nFeatures ({len(self.feature_clusters_)} total):")
        print(
            f"  Multiple memberships: {sum(1 for c in feat_counts if c > 1)} features"
        )
        print(f"  Avg clusters per feature: {np.mean(feat_counts):.2f}")
        print(f"  Max clusters per feature: {max(feat_counts)}")


# Example usage
if __name__ == "__main__":
    # Generate synthetic data with overlapping clusters
    np.random.seed(42)
    I, J, K = 30, 25, 15
    R1, R2, R3 = 4, 3, 3

    # Create tensor with overlapping cluster structure
    X = np.random.rand(I, J, K) * 0.1  # Background noise

    # Add structured patterns (overlapping clusters)
    # Row clusters with overlap
    X[:10, :, :] += 2.0  # Cluster 1
    X[5:15, :, :] += 1.5  # Cluster 2 (overlaps with 1)
    X[20:, :, :] += 1.8  # Cluster 3

    # Column clusters
    X[:, :8, :] += 1.0
    X[:, 10:20, :] += 1.2
    X[:, 15:, :] += 0.8

    # Feature clusters
    X[:, :, :5] += 0.5
    X[:, :, 8:12] += 0.7
    X[:, :, 10:] += 0.6

    X = np.maximum(X, 0)  # Ensure non-negativity

    print(f"Generated tensor shape: {X.shape}")

    # Fit model with multiple cluster memberships
    model = MultiClusterNTTF(
        R1=R1,
        R2=R2,
        R3=R3,
        sparsity_penalty=0.01,  # Encourage sparsity
        membership_threshold=0.3,  # 30% of max for membership
        max_clusters_per_entity=2,  # Max 2 clusters per entity
        init="KMeans",
        num_iter=100,
        verbose=True,
    )

    model.fit(X)

    # Get cluster memberships
    memberships = model.get_cluster_memberships()

    # Print summary
    model.print_cluster_summary()

    # Show some examples
    print(f"\n=== Example Memberships ===")
    print(f"Row 7 belongs to clusters: {memberships['rows'][7]}")
    print(f"Row 12 belongs to clusters: {memberships['rows'][12]}")
    print(f"Column 5 belongs to clusters: {memberships['cols'][5]}")
    print(f"Feature 10 belongs to clusters: {memberships['features'][10]}")

    # Get membership strengths
    strengths = model.get_membership_strengths()
    print(f"\nRow 7 membership strengths: {strengths['rows'][7]}")
    print(f"Row 12 membership strengths: {strengths['rows'][12]}")

    # Reconstruction quality
    X_rec = model.inverse_transform()
    reconstruction_error = model._tensor_frobenius_norm(X - X_rec)
    relative_error = reconstruction_error / model._tensor_frobenius_norm(X)

    print(f"\nReconstruction Quality:")
    print(f"  Absolute error: {reconstruction_error:.6f}")
    print(f"  Relative error: {relative_error:.6f}")
    print(f"  Converged in: {model.n_iter_} iterations")
