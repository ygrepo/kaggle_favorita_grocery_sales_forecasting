import numpy as np
from scipy.sparse import issparse, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
import warnings


class dNMTF(BaseEstimator, TransformerMixin):
    """
    Dual Non-negative Matrix Tri-Factorization (dNMTF)

    Factorizes a matrix X into U, S, V such that X ≈ U @ S @ V.T
    where U, S, V are non-negative matrices.

    Parameters
    ----------
    J : int
        Number of components/rank of factorization
    algorithm : str, default='Frobenius'
        Algorithm to use ('Frobenius', 'KL', 'IS', 'PLTF')
    init : str, default='NMF'
        Initialization method ('NMF', 'ALS-WR', 'Random')
    Beta : float, default=2
        Beta parameter for Beta-divergence
    thr : float, default=1e-10
        Convergence threshold
    num_iter : int, default=100
        Maximum number of iterations
    viz : bool, default=False
        Whether to show progress visualization
    verbose : bool, default=False
        Whether to print verbose output
    """

    def __init__(
        self,
        J,
        algorithm="Frobenius",
        init="NMF",
        Beta=2,
        thr=1e-10,
        num_iter=100,
        viz=False,
        verbose=False,
    ):
        self.J = J
        self.algorithm = algorithm
        self.init = init
        self.Beta = Beta
        self.thr = thr
        self.num_iter = num_iter
        self.viz = viz
        self.verbose = verbose

        # Will be set during fit
        self.U = None
        self.S = None
        self.V = None
        self.RecError = []
        self.RelChange = []
        self.n_iter_ = 0

    def _check_input(self, X):
        """Validate input matrix"""
        X = check_array(X, accept_sparse=["csr", "csc", "coo"])

        if issparse(X):
            X = X.tocsr()

        if np.any(X < 0):
            raise ValueError("Input matrix must be non-negative")

        return X

    def _initialize_factors(self, X):
        """Initialize U, S, V matrices"""
        m, n = X.shape

        if self.init == "Random":
            self.U = np.random.rand(m, self.J)
            self.S = np.random.rand(self.J, self.J)
            self.V = np.random.rand(n, self.J)
        elif self.init == "NMF":
            # Use simple NMF initialization
            from sklearn.decomposition import NMF

            nmf = NMF(n_components=self.J, init="random", random_state=42)
            W = nmf.fit_transform(X)
            H = nmf.components_

            self.U = W
            self.S = np.eye(self.J)
            self.V = H.T
        else:  # ALS-WR or other
            # Simple random initialization
            self.U = np.random.rand(m, self.J)
            self.S = np.random.rand(self.J, self.J)
            self.V = np.random.rand(n, self.J)

        # Ensure non-negativity
        self.U = np.maximum(self.U, 1e-16)
        self.S = np.maximum(self.S, 1e-16)
        self.V = np.maximum(self.V, 1e-16)

    def _reconstruction_error(self, X):
        """Compute reconstruction error"""
        X_rec = self.U @ self.S @ self.V.T

        if self.algorithm == "Frobenius":
            return np.linalg.norm(X - X_rec, "fro") ** 2
        elif self.algorithm == "KL":
            # KL divergence
            return np.sum(X * np.log(X / (X_rec + 1e-16)) - X + X_rec)
        elif self.algorithm == "IS":
            # Itakura-Saito divergence
            return np.sum(
                X / (X_rec + 1e-16) - np.log(X / (X_rec + 1e-16)) - 1
            )
        else:
            # Default to Frobenius
            return np.linalg.norm(X - X_rec, "fro") ** 2

    def _update_frobenius(self, X):
        """Update rules for Frobenius norm"""
        # Update U
        USV = self.U @ self.S @ self.V.T
        numerator = X @ self.V @ self.S.T
        denominator = USV @ self.V @ self.S.T
        self.U *= numerator / (denominator + 1e-16)
        self.U = np.maximum(self.U, 1e-16)

        # Update V
        USV = self.U @ self.S @ self.V.T
        numerator = X.T @ self.U @ self.S
        denominator = USV.T @ self.U @ self.S
        self.V *= numerator / (denominator + 1e-16)
        self.V = np.maximum(self.V, 1e-16)

        # Update S
        numerator = self.U.T @ X @ self.V
        denominator = self.U.T @ self.U @ self.S @ self.V.T @ self.V
        self.S *= numerator / (denominator + 1e-16)
        self.S = np.maximum(self.S, 1e-16)

    def _update_kl(self, X):
        """Update rules for KL divergence"""
        USV = self.U @ self.S @ self.V.T

        # Update U
        numerator = (X / (USV + 1e-16)) @ self.V @ self.S.T
        denominator = np.ones_like(X) @ self.V @ self.S.T
        self.U *= numerator / (denominator + 1e-16)
        self.U = np.maximum(self.U, 1e-16)

        # Update V
        USV = self.U @ self.S @ self.V.T
        numerator = (X / (USV + 1e-16)).T @ self.U @ self.S
        denominator = np.ones_like(X).T @ self.U @ self.S
        self.V *= numerator / (denominator + 1e-16)
        self.V = np.maximum(self.V, 1e-16)

        # Update S
        USV = self.U @ self.S @ self.V.T
        numerator = self.U.T @ (X / (USV + 1e-16)) @ self.V
        denominator = self.U.T @ np.ones_like(X) @ self.V
        self.S *= numerator / (denominator + 1e-16)
        self.S = np.maximum(self.S, 1e-16)

    def _update_is(self, X):
        """Update rules for Itakura-Saito divergence"""
        USV = self.U @ self.S @ self.V.T

        # Update U
        numerator = (X / (USV**2 + 1e-16)) @ self.V @ self.S.T
        denominator = (1 / (USV + 1e-16)) @ self.V @ self.S.T
        self.U *= numerator / (denominator + 1e-16)
        self.U = np.maximum(self.U, 1e-16)

        # Update V
        USV = self.U @ self.S @ self.V.T
        numerator = (X / (USV**2 + 1e-16)).T @ self.U @ self.S
        denominator = (1 / (USV + 1e-16)).T @ self.U @ self.S
        self.V *= numerator / (denominator + 1e-16)
        self.V = np.maximum(self.V, 1e-16)

        # Update S
        USV = self.U @ self.S @ self.V.T
        numerator = self.U.T @ (X / (USV**2 + 1e-16)) @ self.V
        denominator = self.U.T @ (1 / (USV + 1e-16)) @ self.V
        self.S *= numerator / (denominator + 1e-16)
        self.S = np.maximum(self.S, 1e-16)

    def fit(self, X, y=None):
        """
        Fit the dNMTF model to data X

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data matrix
        y : Ignored
            Not used, present for API consistency

        Returns
        -------
        self : object
            Returns the instance itself
        """
        X = self._check_input(X)

        # Initialize factors
        self._initialize_factors(X)

        # Initialize tracking
        self.RecError = []
        self.RelChange = []
        prev_error = float("inf")

        if self.verbose:
            print(f"Starting dNMTF with {self.algorithm} algorithm")

        for iteration in range(self.num_iter):
            # Store previous factors for relative change calculation
            U_prev = self.U.copy()
            S_prev = self.S.copy()
            V_prev = self.V.copy()

            # Update factors based on algorithm
            if self.algorithm == "Frobenius":
                self._update_frobenius(X)
            elif self.algorithm == "KL":
                self._update_kl(X)
            elif self.algorithm == "IS":
                self._update_is(X)
            else:
                self._update_frobenius(X)  # Default

            # Compute reconstruction error
            rec_error = self._reconstruction_error(X)
            self.RecError.append(rec_error)

            # Compute relative change
            rel_change_U = np.linalg.norm(self.U - U_prev, "fro") / (
                np.linalg.norm(U_prev, "fro") + 1e-16
            )
            rel_change_S = np.linalg.norm(self.S - S_prev, "fro") / (
                np.linalg.norm(S_prev, "fro") + 1e-16
            )
            rel_change_V = np.linalg.norm(self.V - V_prev, "fro") / (
                np.linalg.norm(V_prev, "fro") + 1e-16
            )
            rel_change = max(rel_change_U, rel_change_S, rel_change_V)
            self.RelChange.append(rel_change)

            if self.verbose and (iteration + 1) % 10 == 0:
                print(
                    f"Iteration {iteration + 1}: Error = {rec_error:.6f}, RelChange = {rel_change:.6f}"
                )

            # Check convergence
            if rel_change < self.thr:
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break

            prev_error = rec_error

        self.n_iter_ = iteration + 1

        if self.viz:
            self._plot_convergence()

        return self

    def transform(self, X):
        """
        Transform data X using fitted factors

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data

        Returns
        -------
        X_transformed : array, shape (n_samples, n_components)
            Transformed data (U factor)
        """
        if self.U is None:
            raise ValueError("Model must be fitted before transform")

        X = self._check_input(X)

        # For new data, we would need to solve for new U given fixed S, V
        # This is a simplified version - in practice, you might want to
        # solve this optimization problem properly
        return self.U

    def fit_transform(self, X, y=None):
        """
        Fit the model and transform the data

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data
        y : Ignored

        Returns
        -------
        X_transformed : array, shape (n_samples, n_components)
            Transformed data
        """
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X_transformed=None):
        """
        Reconstruct the original data from the factorization

        Returns
        -------
        X_reconstructed : array, shape (n_samples, n_features)
            Reconstructed data
        """
        if self.U is None or self.S is None or self.V is None:
            raise ValueError("Model must be fitted before inverse_transform")

        return self.U @ self.S @ self.V.T

    def _plot_convergence(self):
        """Plot convergence curves"""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.plot(self.RecError)
            ax1.set_title("Reconstruction Error")
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Error")
            ax1.set_yscale("log")

            ax2.plot(self.RelChange)
            ax2.set_title("Relative Change")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Relative Change")
            ax2.set_yscale("log")

            plt.tight_layout()
            plt.show()

        except ImportError:
            warnings.warn("Matplotlib not available for visualization")


# Example usage and test
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    m, n, r = 50, 30, 5

    # Generate ground truth factors
    U_true = np.random.rand(m, r)
    S_true = np.random.rand(r, r)
    V_true = np.random.rand(n, r)

    # Generate noisy data
    X = U_true @ S_true @ V_true.T + 0.1 * np.random.rand(m, n)
    X = np.maximum(X, 0)  # Ensure non-negativity

    # Fit dNMTF
    model = dNMTF(
        J=r, algorithm="Frobenius", num_iter=100, verbose=True, viz=False
    )
    U_est = model.fit_transform(X)

    # Reconstruct
    X_reconstructed = model.inverse_transform()

    # Compute reconstruction error
    reconstruction_error = np.linalg.norm(X - X_reconstructed, "fro")
    print(f"\nFinal reconstruction error: {reconstruction_error:.6f}")
    print(f"Converged in {model.n_iter_} iterations")

    # Print factor shapes
    print(f"\nFactor shapes:")
    print(f"U: {model.U.shape}")
    print(f"S: {model.S.shape}")
    print(f"V: {model.V.shape}")
