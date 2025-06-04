from sklearn.metrics import adjusted_rand_score
import numpy as np
from gdkm import generalized_double_kmeans
from plot_util import visualize_clustered_matrix, visualize_spectral_biclustering
from sklearn.datasets import make_biclusters
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import SpectralBiclustering
import pandas as pd


# Redefine the function now that gdkm is loaded
def run_comparison_on_synthetic_data(random_state=42):
    data, rows, cols = make_biclusters(
        shape=(30, 30), n_clusters=3, noise=0.2, shuffle=True, random_state=random_state
    )

    true_row_labels = np.argmax(rows.T, axis=1)
    true_col_labels = np.argmax(cols.T, axis=1)

    # --- GDKM ---
    U, V_list, _, _ = generalized_double_kmeans(
        data, P=3, Q_list=[3, 3, 3], random_state=random_state
    )
    pred_row_labels_gdkm = np.argmax(U, axis=1)

    pred_col_labels_gdkm = np.zeros(data.shape[1], dtype=int)
    for p, Vp in enumerate(V_list):
        cluster_ids = np.argmax(Vp, axis=1)
        mask = np.any(Vp, axis=1)
        pred_col_labels_gdkm[mask] = cluster_ids[mask] + sum(
            V.shape[1] for V in V_list[:p]
        )

    row_ari_gdkm = adjusted_rand_score(true_row_labels, pred_row_labels_gdkm)
    col_ari_gdkm = adjusted_rand_score(true_col_labels, pred_col_labels_gdkm)
    row_nmi_gdkm = normalized_mutual_info_score(true_row_labels, pred_row_labels_gdkm)
    col_nmi_gdkm = normalized_mutual_info_score(true_col_labels, pred_col_labels_gdkm)

    # --- Spectral Biclustering ---
    sbc = SpectralBiclustering(n_clusters=3, method="log", random_state=random_state)
    sbc.fit(data)
    pred_row_labels_spec = sbc.row_labels_
    pred_col_labels_spec = sbc.column_labels_

    row_ari_spec = adjusted_rand_score(true_row_labels, pred_row_labels_spec)
    col_ari_spec = adjusted_rand_score(true_col_labels, pred_col_labels_spec)
    row_nmi_spec = normalized_mutual_info_score(true_row_labels, pred_row_labels_spec)
    col_nmi_spec = normalized_mutual_info_score(true_col_labels, pred_col_labels_spec)

    # Compile results
    metrics = pd.DataFrame(
        {
            "Metric": ["ARI (rows)", "ARI (columns)", "NMI (rows)", "NMI (columns)"],
            "GDKM": [row_ari_gdkm, col_ari_gdkm, row_nmi_gdkm, col_nmi_gdkm],
            "Spectral Biclustering": [
                row_ari_spec,
                col_ari_spec,
                row_nmi_spec,
                col_nmi_spec,
            ],
        }
    )

    print("GDKM vs Spectral Biclustering Metrics")
    print(metrics)

    # Optionally visualize GDKM results
    visualize_clustered_matrix(
        data,
        U,
        V_list,
        title="GDKM on Synthetic Biclustered Data",
        fn="./output/figures/gdkm.tiff",
    )

    # Optionally visualize Spectral Biclustering results
    visualize_spectral_biclustering(
        data,
        pred_row_labels_spec,
        pred_col_labels_spec,
        title="Spectral Biclustering on Synthetic Biclustered Data",
        fn="./output/figures/spec.tiff",
    )


if __name__ == "__main__":
    run_comparison_on_synthetic_data()
