import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_parafac
from typing import Tuple

# Set NumPy as the backend for consistency
tl.set_backend("numpy")


def fit_ntf_and_get_factors(
    X_3d: np.ndarray, rank: int = 10, n_iter: int = 500, tol: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs Non-Negative CP Decomposition (PARAFAC) on a 3D tensor.

    Args:
        X_3d (I, J, D): Input data tensor (Stores x Items x Features).
        rank: The factorization rank (R), which defines the number of clusters.
        n_iter: Maximum number of iterations.
        tol: Tolerance for convergence.

    Returns:
        (weights, U, V, D) factors, where U and V are the row/column factors.
        U is the Store Membership Factor (I x Rank)
        V is the Item Membership Factor (J x Rank)
    """

    # 1. Prepare Data
    # TensorLy expects the tensor input
    X_tensor = tl.tensor(X_3d)

    # 2. Fit Non-Negative CP Decomposition (PARAFAC)
    # The CP decomposition factors the tensor into a sum of rank-one components.
    # The factors U, V, and D represent the continuous, non-negative cluster memberships
    # for rows (stores), columns (items), and the features dimension.
    weights, factors = non_negative_parafac(
        tensor=X_tensor,
        rank=rank,
        init="random",
        n_iter_max=n_iter,
        tol=tol,
        verbose=True,
        random_state=42,
    )

    # Factors are returned as a list: [Factor_U, Factor_V, Factor_D]
    U_factor = factors[0]  # Store factors (I x Rank)
    V_factor = factors[1]  # Item factors (J x Rank)
    D_factor = factors[2]  # Feature factors (D x Rank)

    return weights, U_factor, V_factor, D_factor