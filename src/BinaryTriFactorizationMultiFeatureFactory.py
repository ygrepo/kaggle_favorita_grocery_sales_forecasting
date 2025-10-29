"""
Factory and Builder for BinaryTriFactorizationMultiFeature.

Provides:
- BTFMultiFeatureBuilder: Picklable builder class for creating instances
- BinaryTriFactorizationMultiFeature.factory(): Class method to create builders
"""

from typing import Optional, Any, Callable, Dict
import numpy as np
from src.utils import get_logger

logger = get_logger(__name__)


class BTFMultiFeatureBuilder:
    """
    Picklable builder class for BinaryTriFactorizationMultiFeature.

    This replaces the local function approach to make it compatible with multiprocessing.
    Allows frozen defaults to be set at builder creation time, with per-call overrides.

    Example:
        >>> builder = BTFMultiFeatureBuilder(
        ...     BinaryTriFactorizationMultiFeature,
        ...     {"alpha": 1e-2, "beta": 0.01, "max_iter": 50}
        ... )
        >>> estimator = builder(
        ...     n_row_clusters=5,
        ...     n_col_clusters=3,
        ...     random_state=42,
        ...     alpha=1e-3  # override frozen default
        ... )
    """

    def __init__(
        self,
        estimator_class: type,
        frozen_kwargs: Dict[str, Any],
    ):
        """
        Initialize the builder.

        Parameters
        ----------
        estimator_class : type
            The BinaryTriFactorizationMultiFeature class (or subclass)
        frozen_kwargs : dict
            Default parameters that are fixed for all instances built by this builder
        """
        self.estimator_class = estimator_class
        self.frozen_kwargs = dict(frozen_kwargs)

    def __call__(
        self,
        n_row_clusters: int,
        n_col_clusters: int,
        *,
        random_state: Optional[int] = None,
        feature_weights: Optional[np.ndarray] = None,
        **overrides: Any,
    ) -> "BinaryTriFactorizationMultiFeature":
        """
        Build an estimator instance with the given parameters.

        Parameters
        ----------
        n_row_clusters : int
            Number of row clusters
        n_col_clusters : int
            Number of column clusters
        random_state : int, optional
            Random seed for reproducibility. If provided, overrides frozen defaults.
        feature_weights : np.ndarray, optional
            Per-feature weights of shape (D,). If provided, overrides frozen defaults.
        **overrides : dict
            Additional parameter overrides for this specific instance

        Returns
        -------
        BinaryTriFactorizationMultiFeature
            A new estimator instance with the specified parameters
        """
        # Start with frozen defaults
        kw: Dict[str, Any] = dict(self.frozen_kwargs)

        # Apply overrides
        kw.update(overrides)

        # Set required parameters
        kw["n_row_clusters"] = int(n_row_clusters)
        kw["n_col_clusters"] = int(n_col_clusters)

        # Handle optional parameters
        if random_state is not None:
            kw["random_state"] = int(random_state)

        if feature_weights is not None:
            kw["feature_weights"] = np.asarray(feature_weights, dtype=np.float64)

        logger.info(
            f"Building {self.estimator_class.__name__} with "
            f"n_row={n_row_clusters}, n_col={n_col_clusters}, "
            f"random_state={kw.get('random_state', 'default')}"
        )

        return self.estimator_class(**kw)

    def __repr__(self) -> str:
        """Return string representation of the builder."""
        frozen_str = ", ".join(
            f"{k}={v!r}" for k, v in self.frozen_kwargs.items()
        )
        return (
            f"{self.__class__.__name__}("
            f"estimator_class={self.estimator_class.__name__}, "
            f"frozen_kwargs={{{frozen_str}}})"
        )


def add_factory_method_to_multi_feature(cls):
    """
    Decorator to add a factory() class method to BinaryTriFactorizationMultiFeature.

    This allows the class to create builder instances with frozen defaults.

    Example:
        >>> @add_factory_method_to_multi_feature
        ... class BinaryTriFactorizationMultiFeature(...):
        ...     pass
        >>>
        >>> builder = BinaryTriFactorizationMultiFeature.factory(
        ...     alpha=1e-2, beta=0.01
        ... )
        >>> estimator = builder(n_row_clusters=5, n_col_clusters=3)
    """

    @classmethod
    def factory(
        cls_inner, **frozen_kwargs
    ) -> Callable[..., "BinaryTriFactorizationMultiFeature"]:
        """
        Create a builder with frozen default parameters.

        Parameters
        ----------
        **frozen_kwargs : dict
            Default parameters to freeze for all instances built by this builder

        Returns
        -------
        BTFMultiFeatureBuilder
            A callable builder that creates estimator instances
        """
        return BTFMultiFeatureBuilder(cls_inner, frozen_kwargs)

    cls.factory = factory
    return cls

