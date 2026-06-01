"""
visualizer.py — Clustering result visualization

Purpose:
  Generate charts to visualize column clusters,
  their distribution across tables, and anti-pattern composition.

  Charts generated:
    1. 2D scatter of clusters (colored by cluster)
    2. 3D scatter of clusters (optional, if dims >= 3)
    3. Silhouette plot
    4. Cluster x table heatmap
    5. Bar chart of types per cluster

Usage:
  viz = Visualizer(output_dir="output/")
  viz.plot_clusters_2d(X_reduced, labels, column_names)
  viz.plot_heatmap(column_table_map, labels)
  viz.save_all(X_reduced, labels, column_table_map, column_metadata)
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from exceptions import VisualizationError

logger = logging.getLogger(__name__)


class Visualizer:
    """Generates visualizations for the clustering pipeline."""

    def __init__(self, output_dir: str = "output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_clusters_2d(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        column_labels: list[str] | None = None,
        title: str = "Clusters de Columnas",
        filename: str = "clusters_2d.png",
    ) -> str:
        """2D scatter plot with colored clusters."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dims = X.shape[1]
        if dims < 2:
            logger.warning("Se necesitan >=2 dimensiones para scatter 2D")
            return ""

        fig, ax = plt.subplots(figsize=(12, 8))
        unique_labels = sorted(set(labels))

        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = f"Cluster {label}" if label != -1 else "Ruido"
            color = "gray" if label == -1 else colors[i]
            ax.scatter(
                X[mask, 0],
                X[mask, 1],
                c=[color],
                label=label_name,
                alpha=0.7,
                s=30,
                edgecolors="black",
                linewidth=0.3,
            )

        if column_labels and len(column_labels) <= 50:
            for i, txt in enumerate(column_labels):
                ax.annotate(txt, (X[i, 0], X[i, 1]), fontsize=6, alpha=0.7)

        ax.set_xlabel("Componente 1")
        ax.set_ylabel("Componente 2")
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        filepath = self.output_dir / filename
        plt.tight_layout()
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Grafico 2D guardado: %s", filepath)
        return str(filepath)

    def plot_clusters_3d(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        title: str = "Clusters de Columnas (3D)",
        filename: str = "clusters_3d.png",
    ) -> str:
        """3D scatter plot (requires >=3 dimensions)."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dims = X.shape[1]
        if dims < 3:
            logger.warning("Se necesitan >=3 dimensiones para scatter 3D")
            return ""

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        unique_labels = sorted(set(labels))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            color = "gray" if label == -1 else colors[i]
            label_name = f"Cluster {label}" if label != -1 else "Ruido"
            ax.scatter(
                X[mask, 0],
                X[mask, 1],
                X[mask, 2],
                c=[color],
                label=label_name,
                alpha=0.7,
                s=30,
            )

        ax.set_xlabel("Comp 1")
        ax.set_ylabel("Comp 2")
        ax.set_zlabel("Comp 3")
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Grafico 3D guardado: %s", filepath)
        return str(filepath)

    def plot_silhouette(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        filename: str = "silhouette.png",
    ) -> str:
        """Silhouette diagram by sample."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import silhouette_samples

        unique = set(labels) - {-1}
        if len(unique) <= 1:
            return ""

        mask = labels != -1
        X_clean = X[mask]
        labels_clean = labels[mask]

        try:
            silhouette_vals = silhouette_samples(X_clean, labels_clean)
        except VisualizationError as e:
            logger.warning("Error generando silhouette: %s", e)
            return ""

        fig, ax = plt.subplots(figsize=(10, 6))
        y_lower = 10
        unique_sorted = sorted(unique)

        for i, label in enumerate(unique_sorted):
            cluster_vals = silhouette_vals[labels_clean == label]
            cluster_vals.sort()
            size = len(cluster_vals)
            y_upper = y_lower + size
            color = plt.cm.tab20(i / len(unique_sorted))
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_vals,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            ax.text(-0.05, y_lower + size / 2, str(label))
            y_lower = y_upper + 10

        avg_sil = float(np.mean(silhouette_vals))
        ax.axvline(x=avg_sil, color="red", linestyle="--", label=f"Media={avg_sil:.3f}")
        ax.set_xlabel("Coeficiente Silhouette")
        ax.set_ylabel("Cluster")
        ax.set_title("Diagrama Silhouette")
        ax.legend()

        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Silhouette guardado: %s", filepath)
        return str(filepath)

    def plot_heatmap(
        self,
        column_table_map: list[str],
        labels: np.ndarray,
        filename: str = "cluster_table_heatmap.png",
    ) -> str:
        """Heatmap of cluster x table distribution."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        data: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for table, label in zip(column_table_map, labels):
            data[table][int(label)] += 1

        df = pd.DataFrame(data).T.fillna(0).astype(int)

        fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 1.5), max(6, len(df) * 0.5)))
        im = ax.imshow(df.values, cmap="YlOrRd", aspect="auto")

        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels([f"C{c}" for c in df.columns], fontsize=8)
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index, fontsize=8)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Tabla")
        ax.set_title("Distribucion Cluster x Tabla")

        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                val = df.values[i, j]
                if val > 0:
                    ax.text(j, i, str(int(val)), ha="center", va="center", fontsize=7)

        fig.colorbar(im, ax=ax, shrink=0.8)

        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Heatmap guardado: %s", filepath)
        return str(filepath)

    def plot_composition(
        self,
        labels: np.ndarray,
        column_metadata: list[Any],
        filename: str = "cluster_composition.png",
    ) -> str:
        """Bar chart of data types per cluster."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        clusters: dict[int, list[str]] = defaultdict(list)
        for label, col in zip(labels, column_metadata):
            clusters[int(label)].append(col.data_type)

        cluster_ids = sorted([c for c in clusters if c != -1])
        if not cluster_ids:
            return ""

        all_types = sorted(set(t for cid in cluster_ids for t in clusters[cid]))
        type_counts: dict[int, Counter] = {cid: Counter(clusters[cid]) for cid in cluster_ids}

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(all_types))
        width = 0.8 / len(cluster_ids)

        for i, cid in enumerate(cluster_ids):
            counts = [type_counts[cid].get(t, 0) for t in all_types]
            offset = (i - len(cluster_ids) / 2 + 0.5) * width
            bars = ax.bar(x + offset, counts, width, label=f"Cluster {cid}")
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1,
                        str(count),
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(all_types, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Tipo de Dato")
        ax.set_ylabel("Cantidad de Columnas")
        ax.set_title("Composicion de Tipos por Cluster")
        ax.legend(fontsize=8)

        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Composicion guardada: %s", filepath)
        return str(filepath)

    def save_all(
        self,
        X_reduced: np.ndarray,
        labels: np.ndarray,
        column_table_map: list[str],
        column_metadata: list[Any],
        column_names: list[str] | None = None,
    ) -> dict[str, str]:
        """
        Generate all visualizations.

        Returns:
            Dict mapping name -> file path.
        """
        files: dict[str, str] = {}

        files["clusters_2d"] = self.plot_clusters_2d(X_reduced, labels, column_names)

        if X_reduced.shape[1] >= 3:
            files["clusters_3d"] = self.plot_clusters_3d(X_reduced, labels)

        files["silhouette"] = self.plot_silhouette(X_reduced, labels)
        files["heatmap"] = self.plot_heatmap(column_table_map, labels)
        files["composition"] = self.plot_composition(labels, column_metadata)

        return files
