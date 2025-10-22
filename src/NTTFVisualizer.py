import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings


class NTTFVisualizer:
    """
    Visualization tools for MultiClusterNTTF results
    """

    def __init__(self, model, X=None):
        """
        Initialize visualizer with fitted model

        Parameters
        ----------
        model : MultiClusterNTTF
            Fitted model
        X : array, shape (I, J, K), optional
            Original tensor data
        """
        self.model = model
        self.X = X
        self.memberships = model.get_cluster_memberships()
        self.strengths = model.get_membership_strengths()

    def plot_membership_heatmaps(self, figsize=(15, 5)):
        """Plot heatmaps of membership strengths"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Row memberships
        im1 = axes[0].imshow(
            self.strengths["rows"], aspect="auto", cmap="Blues"
        )
        axes[0].set_title("Row Cluster Memberships")
        axes[0].set_xlabel("Clusters")
        axes[0].set_ylabel("Rows")
        plt.colorbar(im1, ax=axes[0])

        # Column memberships
        im2 = axes[1].imshow(
            self.strengths["cols"], aspect="auto", cmap="Greens"
        )
        axes[1].set_title("Column Cluster Memberships")
        axes[1].set_xlabel("Clusters")
        axes[1].set_ylabel("Columns")
        plt.colorbar(im2, ax=axes[1])

        # Feature memberships
        im3 = axes[2].imshow(
            self.strengths["features"], aspect="auto", cmap="Reds"
        )
        axes[2].set_title("Feature Cluster Memberships")
        axes[2].set_xlabel("Clusters")
        axes[2].set_ylabel("Features")
        plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()
        return fig

    def plot_cluster_overlap_matrix(self, mode="rows", figsize=(8, 6)):
        """Plot overlap matrix showing co-membership patterns"""
        if mode == "rows":
            memberships = self.memberships["rows"]
            title = "Row Co-membership Matrix"
        elif mode == "cols":
            memberships = self.memberships["cols"]
            title = "Column Co-membership Matrix"
        elif mode == "features":
            memberships = self.memberships["features"]
            title = "Feature Co-membership Matrix"
        else:
            raise ValueError("mode must be 'rows', 'cols', or 'features'")

        n_entities = len(memberships)
        overlap_matrix = np.zeros((n_entities, n_entities))

        # Compute overlap (Jaccard similarity)
        for i in range(n_entities):
            for j in range(n_entities):
                set_i = set(memberships[i])
                set_j = set(memberships[j])
                if len(set_i.union(set_j)) > 0:
                    overlap_matrix[i, j] = len(
                        set_i.intersection(set_j)
                    ) / len(set_i.union(set_j))

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(overlap_matrix, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Entity Index")
        ax.set_ylabel("Entity Index")
        plt.colorbar(im, ax=ax, label="Jaccard Similarity")

        return fig

    def plot_cluster_sizes_and_overlaps(self, figsize=(15, 5)):
        """Plot cluster sizes and overlap statistics"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        modes = ["rows", "cols", "features"]
        colors = ["blue", "green", "red"]

        for idx, (mode, color) in enumerate(zip(modes, colors)):
            memberships = self.memberships[mode]

            # Cluster sizes
            max_clusters = (
                max(len(m) for m in memberships) if memberships else 1
            )
            cluster_counts = [0] * (max_clusters + 1)

            for member_list in memberships:
                cluster_counts[len(member_list)] += 1

            axes[idx].bar(
                range(len(cluster_counts)),
                cluster_counts,
                color=color,
                alpha=0.7,
            )
            axes[idx].set_title(f"{mode.capitalize()} - Clusters per Entity")
            axes[idx].set_xlabel("Number of Clusters")
            axes[idx].set_ylabel("Number of Entities")
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_tensor_slices_with_clusters(
        self, feature_indices=None, figsize=(15, 10)
    ):
        """Plot tensor slices with cluster boundaries highlighted"""
        if self.X is None:
            raise ValueError(
                "Original tensor X must be provided to visualizer"
            )

        I, J, K = self.X.shape

        if feature_indices is None:
            # Select a few representative features
            feature_indices = np.linspace(0, K - 1, min(6, K), dtype=int)

        n_features = len(feature_indices)
        cols = min(3, n_features)
        rows = (n_features + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_features == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        # Get cluster assignments (primary cluster for each entity)
        row_primary = [
            clusters[0] if clusters else 0
            for clusters in self.memberships["rows"]
        ]
        col_primary = [
            clusters[0] if clusters else 0
            for clusters in self.memberships["cols"]
        ]

        for idx, k in enumerate(feature_indices):
            row_idx = idx // cols
            col_idx = idx % cols

            if rows > 1:
                ax = axes[row_idx, col_idx]
            else:
                ax = axes[col_idx] if cols > 1 else axes

            # Plot tensor slice
            im = ax.imshow(self.X[:, :, k], aspect="auto", cmap="viridis")

            # Add cluster boundaries
            self._add_cluster_boundaries(ax, row_primary, col_primary, I, J)

            ax.set_title(
                f'Feature {k} (Cluster {self.memberships["features"][k]})'
            )
            ax.set_xlabel("Columns")
            ax.set_ylabel("Rows")

            plt.colorbar(im, ax=ax)

        # Hide empty subplots
        for idx in range(n_features, rows * cols):
            row_idx = idx // cols
            col_idx = idx % cols
            if rows > 1:
                axes[row_idx, col_idx].set_visible(False)
            elif cols > 1:
                axes[col_idx].set_visible(False)

        plt.tight_layout()
        return fig

    def _add_cluster_boundaries(self, ax, row_clusters, col_clusters, I, J):
        """Add cluster boundary lines to plot"""
        # Sort indices by cluster
        row_sorted_idx = np.argsort(row_clusters)
        col_sorted_idx = np.argsort(col_clusters)

        # Find cluster boundaries
        row_boundaries = []
        col_boundaries = []

        if len(row_sorted_idx) > 0:
            current_row_cluster = row_clusters[row_sorted_idx[0]]
            for i, idx in enumerate(row_sorted_idx[1:], 1):
                if row_clusters[idx] != current_row_cluster:
                    row_boundaries.append(i - 0.5)
                    current_row_cluster = row_clusters[idx]

        if len(col_sorted_idx) > 0:
            current_col_cluster = col_clusters[col_sorted_idx[0]]
            for j, idx in enumerate(col_sorted_idx[1:], 1):
                if col_clusters[idx] != current_col_cluster:
                    col_boundaries.append(j - 0.5)
                    current_col_cluster = col_clusters[idx]

        # Draw boundaries
        for boundary in row_boundaries:
            ax.axhline(y=boundary, color="red", linewidth=2, alpha=0.7)

        for boundary in col_boundaries:
            ax.axvline(x=boundary, color="red", linewidth=2, alpha=0.7)

    def plot_cluster_network(self, mode="rows", figsize=(12, 8)):
        """Plot network graph showing cluster relationships"""
        if mode == "rows":
            memberships = self.memberships["rows"]
            strengths = self.strengths["rows"]
            title = "Row Cluster Network"
        elif mode == "cols":
            memberships = self.memberships["cols"]
            strengths = self.strengths["cols"]
            title = "Column Cluster Network"
        elif mode == "features":
            memberships = self.memberships["features"]
            strengths = self.strengths["features"]
            title = "Feature Cluster Network"

        # Create bipartite graph: entities and clusters
        G = nx.Graph()

        # Add entity nodes
        entity_nodes = [f"E{i}" for i in range(len(memberships))]
        cluster_nodes = [f"C{j}" for j in range(strengths.shape[1])]

        G.add_nodes_from(entity_nodes, bipartite=0)
        G.add_nodes_from(cluster_nodes, bipartite=1)

        # Add edges with weights
        for i, entity_clusters in enumerate(memberships):
            for cluster in entity_clusters:
                weight = strengths[i, cluster]
                G.add_edge(f"E{i}", f"C{cluster}", weight=weight)

        # Layout
        pos = {}
        # Position entities on left, clusters on right
        if len(entity_nodes) > 1:
            entity_y = np.linspace(0, 1, len(entity_nodes))
        else:
            entity_y = [0.5]

        if len(cluster_nodes) > 1:
            cluster_y = np.linspace(0, 1, len(cluster_nodes))
        else:
            cluster_y = [0.5]

        for i, node in enumerate(entity_nodes):
            pos[node] = (0, entity_y[i])
        for i, node in enumerate(cluster_nodes):
            pos[node] = (1, cluster_y[i])

        fig, ax = plt.subplots(figsize=figsize)

        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=entity_nodes,
            node_color="lightblue",
            node_size=100,
            ax=ax,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=cluster_nodes,
            node_color="lightcoral",
            node_size=200,
            ax=ax,
        )

        # Draw edges with varying thickness based on strength
        edges = G.edges()
        weights = [G[u][v]["weight"] for u, v in edges]
        if weights:
            nx.draw_networkx_edges(
                G, pos, width=[w * 3 for w in weights], alpha=0.6, ax=ax
            )

        # Labels
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

        ax.set_title(title)
        ax.axis("off")

        return fig

    def plot_dimensionality_reduction(self, method="tsne", figsize=(15, 5)):
        """Plot 2D embeddings of entities colored by clusters"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        modes = ["rows", "cols", "features"]
        factors = [self.model.A, self.model.B, self.model.C]

        for idx, (mode, factor) in enumerate(zip(modes, factors)):
            # Apply dimensionality reduction
            if factor.shape[0] < 2:
                # Not enough points for dimensionality reduction
                embedding = np.random.rand(factor.shape[0], 2)
            elif method == "tsne":
                if factor.shape[0] > 2:  # Need at least 3 points for t-SNE
                    perplexity = min(30, max(1, factor.shape[0] - 1))
                    reducer = TSNE(
                        n_components=2, random_state=42, perplexity=perplexity
                    )
                    embedding = reducer.fit_transform(factor)
                else:
                    embedding = np.random.rand(factor.shape[0], 2)
            elif method == "pca":
                n_components = min(2, factor.shape[1])
                reducer = PCA(n_components=n_components, random_state=42)
                embedding = reducer.fit_transform(factor)
                if embedding.shape[1] == 1:
                    # Add second dimension if only 1 component
                    embedding = np.column_stack(
                        [embedding, np.zeros(embedding.shape[0])]
                    )
            else:
                raise ValueError("method must be 'tsne' or 'pca'")

            # Color by primary cluster
            memberships = self.memberships[mode]
            colors = [
                clusters[0] if clusters else 0 for clusters in memberships
            ]

            scatter = axes[idx].scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=colors,
                cmap="tab10",
                s=50,
                alpha=0.7,
            )

            # Highlight multi-cluster entities
            multi_cluster_mask = [
                len(clusters) > 1 for clusters in memberships
            ]
            if any(multi_cluster_mask):
                multi_embedding = embedding[multi_cluster_mask]
                axes[idx].scatter(
                    multi_embedding[:, 0],
                    multi_embedding[:, 1],
                    s=100,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=2,
                    label="Multi-cluster",
                )
                axes[idx].legend()

            axes[idx].set_title(
                f"{mode.capitalize()} - {method.upper()} Embedding"
            )
            axes[idx].set_xlabel(f"{method.upper()}1")
            axes[idx].set_ylabel(f"{method.upper()}2")

            # Add colorbar
            plt.colorbar(scatter, ax=axes[idx], label="Primary Cluster")

        plt.tight_layout()
        return fig

    def plot_reconstruction_quality(self, figsize=(12, 8)):
        """Plot reconstruction quality analysis"""
        if self.X is None:
            raise ValueError("Original tensor X must be provided")

        X_rec = self.model.inverse_transform()

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Overall reconstruction scatter
        axes[0, 0].scatter(self.X.flatten(), X_rec.flatten(), alpha=0.5, s=1)
        axes[0, 0].plot(
            [self.X.min(), self.X.max()], [self.X.min(), self.X.max()], "r--"
        )
        axes[0, 0].set_xlabel("Original Values")
        axes[0, 0].set_ylabel("Reconstructed Values")
        axes[0, 0].set_title("Reconstruction Scatter Plot")

        # 2. Error by feature
        feature_errors = []
        for k in range(self.X.shape[2]):
            error = np.linalg.norm(self.X[:, :, k] - X_rec[:, :, k])
            feature_errors.append(error)

        axes[0, 1].bar(range(len(feature_errors)), feature_errors)
        axes[0, 1].set_xlabel("Feature Index")
        axes[0, 1].set_ylabel("Reconstruction Error")
        axes[0, 1].set_title("Error by Feature")

        # 3. Residual heatmap (first feature) - FIXED
        residual = self.X[:, :, 0] - X_rec[:, :, 0]

        # Create symmetric colormap centered at 0
        vmax = max(abs(residual.min()), abs(residual.max()))
        vmin = -vmax

        im = axes[1, 0].imshow(residual, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[1, 0].set_title("Residuals (Feature 0)")
        axes[1, 0].set_xlabel("Columns")
        axes[1, 0].set_ylabel("Rows")
        plt.colorbar(im, ax=axes[1, 0])

        # 4. Training convergence
        if hasattr(self.model, "RecError") and len(self.model.RecError) > 0:
            axes[1, 1].plot(self.model.RecError, label="Reconstruction Error")
        if hasattr(self.model, "RelChange") and len(self.model.RelChange) > 0:
            axes[1, 1].plot(self.model.RelChange, label="Relative Change")

        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Value")
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_title("Training Convergence")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_comprehensive_report(self, save_path=None):
        """Create a comprehensive visualization report"""
        figs = []

        print("Creating comprehensive visualization report...")

        # 1. R-style convergence curves
        try:
            fig_conv = self.plot_r_style_convergence()
            figs.append(("r_style_convergence", fig_conv))
            print("✓ Created R-style convergence curves")
        except Exception as e:
            print(f"✗ Could not create R-style convergence curves: {e}")

        # 2. R-style factor histograms
        try:
            fig_hist = self.plot_r_style_histograms()
            figs.append(("r_style_histograms", fig_hist))
            print("✓ Created R-style factor histograms")
        except Exception as e:
            print(f"✗ Could not create R-style factor histograms: {e}")

        # 3. Detailed factor distributions
        try:
            fig_detailed = self.plot_factor_distributions_detailed()
            figs.append(("detailed_factor_distributions", fig_detailed))
            print("✓ Created detailed factor distributions")
        except Exception as e:
            print(f"✗ Could not create detailed factor distributions: {e}")

        # 1. Membership heatmaps
        try:
            fig1 = self.plot_membership_heatmaps()
            figs.append(("membership_heatmaps", fig1))
            print("✓ Created membership heatmaps")
        except Exception as e:
            print(f"✗ Could not create membership heatmaps: {e}")

        # 2. Cluster sizes and overlaps
        try:
            fig2 = self.plot_cluster_sizes_and_overlaps()
            figs.append(("cluster_sizes", fig2))
            print("✓ Created cluster size plots")
        except Exception as e:
            print(f"✗ Could not create cluster size plots: {e}")

        # 3. Dimensionality reduction
        try:
            fig3 = self.plot_dimensionality_reduction(method="pca")
            figs.append(("pca_embedding", fig3))
            print("✓ Created PCA embedding")
        except Exception as e:
            print(f"✗ Could not create PCA embedding: {e}")

        # 4. Overlap matrices
        for mode in ["rows", "cols", "features"]:
            try:
                fig = self.plot_cluster_overlap_matrix(mode=mode)
                figs.append((f"overlap_matrix_{mode}", fig))
                print(f"✓ Created {mode} overlap matrix")
            except Exception as e:
                print(f"✗ Could not create {mode} overlap matrix: {e}")

        # 5. Network graphs
        for mode in ["rows", "cols", "features"]:
            try:
                fig = self.plot_cluster_network(mode=mode)
                figs.append((f"network_{mode}", fig))
                print(f"✓ Created {mode} network graph")
            except Exception as e:
                print(f"✗ Could not create {mode} network graph: {e}")

        # 6. Tensor slices (if data available)
        if self.X is not None:
            try:
                fig5 = self.plot_tensor_slices_with_clusters()
                figs.append(("tensor_slices", fig5))
                print("✓ Created tensor slices")
            except Exception as e:
                print(f"✗ Could not create tensor slices: {e}")

            try:
                fig6 = self.plot_reconstruction_quality()
                figs.append(("reconstruction_quality", fig6))
                print("✓ Created reconstruction quality plot")
            except Exception as e:
                print(f"✗ Could not create reconstruction quality plot: {e}")

        # Save figures if path provided
        if save_path:
            import os

            os.makedirs(save_path, exist_ok=True)
            for name, fig in figs:
                try:
                    fig.savefig(
                        f"{save_path}/{name}.png", dpi=150, bbox_inches="tight"
                    )
                    print(f"✓ Saved {name}.png")
                except Exception as e:
                    print(f"✗ Could not save {name}.png: {e}")

        print(f"\nReport complete! Generated {len(figs)} visualizations.")
        return figs

    def plot_convergence_curves(self, figsize=(12, 5)):
        """
        Plot convergence curves for reconstruction error and relative change
        Similar to the R dcTensor package style
        """
        if not hasattr(self.model, "RecError") or not hasattr(
            self.model, "RelChange"
        ):
            raise ValueError(
                "Model must have RecError and RelChange attributes from training"
            )

        if len(self.model.RecError) == 0 or len(self.model.RelChange) == 0:
            raise ValueError("No convergence data available")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Reconstruction Error plot
        iterations = range(len(self.model.RecError))
        ax1.plot(
            iterations,
            self.model.RecError,
            "o-",
            markersize=4,
            linewidth=1,
            markerfacecolor="white",
            markeredgecolor="black",
            color="black",
        )
        ax1.set_xlabel("Index", fontsize=12)
        ax1.set_ylabel("log10(out_SBMTF$RecError)", fontsize=12)
        ax1.set_title("Reconstruction Error", fontsize=14, fontweight="bold")
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # Relative Change plot
        iterations = range(len(self.model.RelChange))
        ax2.plot(
            iterations,
            self.model.RelChange,
            "o-",
            markersize=4,
            linewidth=1,
            markerfacecolor="white",
            markeredgecolor="black",
            color="black",
        )
        ax2.set_xlabel("Index", fontsize=12)
        ax2.set_ylabel("log10(out_SBMTF$RelChange)", fontsize=12)
        ax2.set_title("Relative Change", fontsize=14, fontweight="bold")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        plt.tight_layout()
        return fig

    def plot_factor_histograms(self, figsize=(15, 5), bins=30):
        """
        Plot histograms of the factor matrices (A, S, V equivalent to U, S, V)
        Similar to the R dcTensor package style
        """
        if any(f is None for f in [self.model.A, self.model.S, self.model.C]):
            raise ValueError("Model must be fitted first")

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Factor A (rows) - equivalent to U
        axes[0].hist(
            self.model.A.flatten(),
            bins=bins,
            color="lightgray",
            edgecolor="black",
            alpha=0.7,
        )
        axes[0].set_xlabel("Factor A (Rows)", fontsize=12)
        axes[0].set_ylabel("Frequency", fontsize=12)
        axes[0].set_title(
            "Histogram of Factor A", fontsize=14, fontweight="bold"
        )
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)
        axes[0].grid(True, alpha=0.3, axis="y")

        # Core tensor S - equivalent to S
        axes[1].hist(
            self.model.S.flatten(),
            bins=bins,
            color="lightgray",
            edgecolor="black",
            alpha=0.7,
        )
        axes[1].set_xlabel("Core Tensor S", fontsize=12)
        axes[1].set_ylabel("Frequency", fontsize=12)
        axes[1].set_title(
            "Histogram of Core Tensor S", fontsize=14, fontweight="bold"
        )
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].grid(True, alpha=0.3, axis="y")

        # Factor C (features) - equivalent to V
        axes[2].hist(
            self.model.C.flatten(),
            bins=bins,
            color="lightgray",
            edgecolor="black",
            alpha=0.7,
        )
        axes[2].set_xlabel("Factor C (Features)", fontsize=12)
        axes[2].set_ylabel("Frequency", fontsize=12)
        axes[2].set_title(
            "Histogram of Factor C", fontsize=14, fontweight="bold"
        )
        axes[2].spines["top"].set_visible(False)
        axes[2].spines["right"].set_visible(False)
        axes[2].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig

    def plot_factor_distributions_detailed(self, figsize=(18, 12), bins=50):
        """
        Plot detailed distributions of all factors including factor B
        Extended version with more comprehensive analysis
        """
        if any(
            f is None
            for f in [self.model.A, self.model.B, self.model.C, self.model.S]
        ):
            raise ValueError("Model must be fitted first")

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        factors = {
            "Factor A (Rows)": self.model.A,
            "Factor B (Columns)": self.model.B,
            "Factor C (Features)": self.model.C,
            "Core Tensor S": self.model.S,
        }

        # Plot histograms for first 4 factors
        factor_names = list(factors.keys())
        positions = [(0, 0), (0, 1), (0, 2), (1, 0)]

        for i, (name, factor) in enumerate(factors.items()):
            if i >= 4:
                break

            row, col = positions[i]
            data = factor.flatten()

            # Histogram
            n, bins_edges, patches = axes[row, col].hist(
                data,
                bins=bins,
                color="lightgray",
                edgecolor="black",
                alpha=0.7,
                density=True,
            )

            # Add statistics text
            mean_val = np.mean(data)
            std_val = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)

            stats_text = f"Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMin: {min_val:.3f}\nMax: {max_val:.3f}"
            axes[row, col].text(
                0.65,
                0.95,
                stats_text,
                transform=axes[row, col].transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontsize=9,
            )

            axes[row, col].set_xlabel(name, fontsize=12)
            axes[row, col].set_ylabel("Density", fontsize=12)
            axes[row, col].set_title(
                f"Distribution of {name}", fontsize=14, fontweight="bold"
            )
            axes[row, col].spines["top"].set_visible(False)
            axes[row, col].spines["right"].set_visible(False)
            axes[row, col].grid(True, alpha=0.3, axis="y")

        # Factor sparsity analysis
        sparsity_threshold = 1e-3
        sparsities = {}
        for name, factor in factors.items():
            sparse_ratio = np.sum(factor < sparsity_threshold) / factor.size
            sparsities[name] = sparse_ratio

        # Plot sparsity
        names = list(sparsities.keys())
        values = list(sparsities.values())

        axes[1, 1].bar(
            range(len(names)),
            values,
            color=["lightblue", "lightgreen", "lightcoral", "lightyellow"],
            edgecolor="black",
            alpha=0.7,
        )
        axes[1, 1].set_xticks(range(len(names)))
        axes[1, 1].set_xticklabels(
            [name.split()[1] for name in names], rotation=45
        )
        axes[1, 1].set_ylabel("Sparsity Ratio", fontsize=12)
        axes[1, 1].set_title(
            "Factor Sparsity Analysis", fontsize=14, fontweight="bold"
        )
        axes[1, 1].spines["top"].set_visible(False)
        axes[1, 1].spines["right"].set_visible(False)
        axes[1, 1].grid(True, alpha=0.3, axis="y")

        # Factor norms comparison
        norms = {}
        for name, factor in factors.items():
            if factor.ndim == 2:
                norm_val = np.linalg.norm(factor, "fro")
            else:  # 3D tensor
                norm_val = np.sqrt(np.sum(factor**2))
            norms[name] = norm_val

        names = list(norms.keys())
        values = list(norms.values())

        axes[1, 2].bar(
            range(len(names)),
            values,
            color=["lightblue", "lightgreen", "lightcoral", "lightyellow"],
            edgecolor="black",
            alpha=0.7,
        )
        axes[1, 2].set_xticks(range(len(names)))
        axes[1, 2].set_xticklabels(
            [name.split()[1] for name in names], rotation=45
        )
        axes[1, 2].set_ylabel("Frobenius Norm", fontsize=12)
        axes[1, 2].set_title(
            "Factor Norms Comparison", fontsize=14, fontweight="bold"
        )
        axes[1, 2].spines["top"].set_visible(False)
        axes[1, 2].spines["right"].set_visible(False)
        axes[1, 2].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return fig

    def plot_r_style_convergence(self, figsize=(12, 5)):
        """
        Plot convergence in exact R dcTensor package style
        """
        if not hasattr(self.model, "RecError") or not hasattr(
            self.model, "RelChange"
        ):
            raise ValueError(
                "Model must have RecError and RelChange attributes"
            )

        # Set R-like style
        plt.style.use("default")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.patch.set_facecolor("white")

        # Reconstruction Error - R style
        iterations = range(len(self.model.RecError))
        ax1.plot(
            iterations,
            self.model.RecError,
            "o",
            markersize=3,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.5,
            linestyle="none",
        )

        ax1.set_xlabel("Index", fontsize=11)
        ax1.set_ylabel("log10(out_SBMTF$RecError)", fontsize=11)
        ax1.set_title("Reconstruction Error", fontsize=12, pad=20)
        ax1.set_yscale("log")

        # R-style formatting
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_linewidth(0.5)
        ax1.spines["bottom"].set_linewidth(0.5)
        ax1.tick_params(axis="both", which="major", labelsize=10, width=0.5)
        ax1.grid(False)
        ax1.set_facecolor("white")

        # Relative Change - R style
        iterations = range(len(self.model.RelChange))
        ax2.plot(
            iterations,
            self.model.RelChange,
            "o",
            markersize=3,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.5,
            linestyle="none",
        )

        ax2.set_xlabel("Index", fontsize=11)
        ax2.set_ylabel("log10(out_SBMTF$RelChange)", fontsize=11)
        ax2.set_title("Relative Change", fontsize=12, pad=20)
        ax2.set_yscale("log")

        # R-style formatting
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_linewidth(0.5)
        ax2.spines["bottom"].set_linewidth(0.5)
        ax2.tick_params(axis="both", which="major", labelsize=10, width=0.5)
        ax2.grid(False)
        ax2.set_facecolor("white")

        plt.tight_layout()
        return fig

    def plot_r_style_histograms(self, figsize=(15, 5), bins=30):
        """
        Plot factor histograms in exact R dcTensor package style
        """
        if any(f is None for f in [self.model.A, self.model.S, self.model.C]):
            raise ValueError("Model must be fitted first")

        # Set R-like style
        plt.style.use("default")

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.patch.set_facecolor("white")

        # Factor A (U equivalent)
        axes[0].hist(
            self.model.A.flatten(),
            bins=bins,
            color="lightgray",
            edgecolor="black",
            linewidth=0.5,
            alpha=1.0,
        )
        axes[0].set_xlabel("out_SBMTF$U", fontsize=11)
        axes[0].set_ylabel("Frequency", fontsize=11)
        axes[0].set_title("Histogram of out_SBMTF$U", fontsize=12, pad=20)

        # R-style formatting
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)
        axes[0].spines["left"].set_linewidth(0.5)
        axes[0].spines["bottom"].set_linewidth(0.5)
        axes[0].tick_params(
            axis="both", which="major", labelsize=10, width=0.5
        )
        axes[0].grid(False)
        axes[0].set_facecolor("white")

        # Core tensor S
        axes[1].hist(
            self.model.S.flatten(),
            bins=bins,
            color="lightgray",
            edgecolor="black",
            linewidth=0.5,
            alpha=1.0,
        )
        axes[1].set_xlabel("out_SBMTF$S", fontsize=11)
        axes[1].set_ylabel("Frequency", fontsize=11)
        axes[1].set_title("Histogram of out_SBMTF$S", fontsize=12, pad=20)

        # R-style formatting
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].spines["left"].set_linewidth(0.5)
        axes[1].spines["bottom"].set_linewidth(0.5)
        axes[1].tick_params(
            axis="both", which="major", labelsize=10, width=0.5
        )
        axes[1].grid(False)
        axes[1].set_facecolor("white")

        # Factor C (V equivalent)
        axes[2].hist(
            self.model.C.flatten(),
            bins=bins,
            color="lightgray",
            edgecolor="black",
            linewidth=0.5,
            alpha=1.0,
        )
        axes[2].set_xlabel("out_SBMTF$V", fontsize=11)
        axes[2].set_ylabel("Frequency", fontsize=11)
        axes[2].set_title("Histogram of out_SBMTF$V", fontsize=12, pad=20)

        # R-style formatting
        axes[2].spines["top"].set_visible(False)
        axes[2].spines["right"].set_visible(False)
        axes[2].spines["left"].set_linewidth(0.5)
        axes[2].spines["bottom"].set_linewidth(0.5)
        axes[2].tick_params(
            axis="both", which="major", labelsize=10, width=0.5
        )
        axes[2].grid(False)
        axes[2].set_facecolor("white")

        plt.tight_layout()
        return fig


# Enhanced example with visualization
if __name__ == "__main__":
    # Import our main class - adjust path as needed
    import sys
    import os

    sys.path.append(os.path.dirname(__file__))

    try:
        from MultiClusterNTTF import MultiClusterNTTF
    except ImportError:
        print(
            "MultiClusterNTTF not found. Please ensure the file is in the same directory."
        )
        sys.exit(1)

    # Generate synthetic data
    np.random.seed(42)
    I, J, K = 30, 25, 15
    R1, R2, R3 = 4, 3, 3

    # Create tensor with clear cluster structure
    X = np.random.rand(I, J, K) * 0.1

    # Add structured patterns
    X[:10, :, :] += 2.0
    X[5:15, :, :] += 1.5
    X[20:, :, :] += 1.8
    X[:, :8, :] += 1.0
    X[:, 10:20, :] += 1.2
    X[:, 15:, :] += 0.8
    X[:, :, :5] += 0.5
    X[:, :, 8:12] += 0.7
    X[:, :, 10:] += 0.6

    X = np.maximum(X, 0)

    print(f"Generated tensor shape: {X.shape}")

    # Fit model
    model = MultiClusterNTTF(
        R1=R1,
        R2=R2,
        R3=R3,
        sparsity_penalty=0.01,
        membership_threshold=0.3,
        max_clusters_per_entity=2,
        init="KMeans",
        num_iter=50,
        verbose=True,
    )

    model.fit(X)

    # Create visualizer
    visualizer = NTTFVisualizer(model, X)

    # Generate all visualizations
    print("\n" + "=" * 50)
    print("GENERATING VISUALIZATIONS")
    print("=" * 50)

    # Individual plots
    try:
        fig1 = visualizer.plot_membership_heatmaps()
        plt.show()
    except Exception as e:
        print(f"Error in membership heatmaps: {e}")

    try:
        fig2 = visualizer.plot_cluster_sizes_and_overlaps()
        plt.show()
    except Exception as e:
        print(f"Error in cluster sizes: {e}")

    try:
        fig3 = visualizer.plot_dimensionality_reduction(method="pca")
        plt.show()
    except Exception as e:
        print(f"Error in dimensionality reduction: {e}")

    try:
        fig4 = visualizer.plot_tensor_slices_with_clusters()
        plt.show()
    except Exception as e:
        print(f"Error in tensor slices: {e}")

    try:
        fig5 = visualizer.plot_reconstruction_quality()
        plt.show()
    except Exception as e:
        print(f"Error in reconstruction quality: {e}")

    # Comprehensive report
    try:
        visualizer.create_comprehensive_report(save_path="nttf_results")
    except Exception as e:
        print(f"Error creating comprehensive report: {e}")

    print("\nVisualization complete!")
