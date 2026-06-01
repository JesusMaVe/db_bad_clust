"""
phase2_orchestrator.py — Phase 2 full pipeline orchestrator

Purpose:
  Executes the complete ML pipeline on DB schemas:
  Connection -> Extraction -> Preprocessing -> BERT Embeddings
    -> Structural Encoding -> Feature Vectors -> Dimensionality Reduction
    -> Clustering -> Evaluation -> Recommendations -> Visualization

  Supports automatic hyperparameter tuning with --tune.

Usage:
  cd scripts && python phase2_orchestrator.py
  cd scripts && python phase2_orchestrator.py --skip-bert
  cd scripts && python phase2_orchestrator.py --tune
  cd scripts && python phase2_orchestrator.py --alpha 0.4 --beta 0.3 --gamma 0.3
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from bert_embedder import BERTEmbedder
from cluster_engine import ClusterEngine
from db_connector import OracleConnector
from dimensionality_reducer import DimensionalityReducer
from evaluator import Evaluator
from exceptions import BadDBError, DatabaseError, EmbeddingError
from feature_builder import FeatureBuilder
from ground_truth import build_ground_truth_map, get_ground_truth
from recommender import Recommender
from schema_extractor import ColumnMetadata, DatabaseSchema, SchemaExtractor
from structural_encoder import StructuralEncoder
from text_preprocessor import TextPreprocessor
from visualizer import Visualizer

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Phase 2 pipeline."""
    parser = argparse.ArgumentParser(description="Phase 2: ML Pipeline on DB Schemas")
    parser.add_argument(
        "--skip-bert",
        action="store_true",
        help="Skip BERT (use synthetic embeddings)",
    )
    parser.add_argument(
        "--method",
        default="pca",
        choices=["pca", "umap", "svd", "tsne"],
        help="Dimensionality reduction method",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Target dimensions after reduction (default: auto, 80%% variance)",
    )
    parser.add_argument(
        "--cluster-method",
        default="kmeans",
        choices=["kmeans", "dbscan", "agglomerative", "meanshift", "hdbscan"],
        help="Clustering algorithm",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Number of clusters (default: auto via silhouette)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="BERT semantic weight (default: 0.40)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Data type weight (default: 0.30)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Constraint weight (default: 0.25)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Statistical weight (data_length) (default: 0.05)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Auto-tune hyperparameters (alpha, beta, gamma, delta, n_components, n_clusters)",
    )
    parser.add_argument(
        "--metric",
        default="euclidean",
        choices=["euclidean", "cosine"],
        help="Distance metric for DBSCAN (default: euclidean)",
    )
    parser.add_argument(
        "--dbscan-eps", default=None, type=str,
        help="DBSCAN eps value (default: 0.5, use 'auto' for k-distance estimation)",
    )
    parser.add_argument(
        "--dbscan-min-pts",
        type=int,
        default=None,
        help="DBSCAN min_samples (default: auto, 5%% of data)",
    )
    parser.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        default=5,
        help="HDBSCAN min_cluster_size (default: 5)",
    )
    parser.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=None,
        help="HDBSCAN min_samples (default: auto)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate clustering against anti-pattern ground truth (ARI/NMI)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for charts and reports (default: ../output/run_<timestamp>)",
    )
    return parser.parse_args()


def run_pipeline(
    e_text: np.ndarray,
    e_type: np.ndarray,
    e_rest: np.ndarray,
    e_stat: np.ndarray,
    n_components: int,
    n_clusters: int,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    method: str,
    cluster_method: str,
    skip_bert: bool,
) -> dict[str, Any]:
    """Run dimensionality reduction, clustering, and evaluation with given parameters.

    Args:
        e_text: BERT/synthetic embeddings, shape (N, 768).
        e_type: One-hot encoded data types, shape (N, n_types).
        e_rest: Binary constraint encodings, shape (N, 5).
        e_stat: Statistical features, shape (N, n_stats).
        n_components: Target dimensions after reduction.
        n_clusters: Number of clusters.
        alpha: BERT semantic weight.
        beta: Data type weight.
        gamma: Constraint weight.
        delta: Statistical feature weight.
        method: Dimensionality reduction method (pca, umap, svd, tsne).
        cluster_method: Clustering algorithm (kmeans, dbscan, agglomerative).
        skip_bert: Whether BERT was skipped (kept for interface consistency).

    Returns:
        Dict with keys: labels, X_reduced, metrics, params.
    """
    builder: FeatureBuilder = FeatureBuilder(alpha=alpha, beta=beta, gamma=gamma, delta=delta)
    phi_w: np.ndarray = builder.build(e_text, e_type, e_rest, e_stat)

    reducer: DimensionalityReducer = DimensionalityReducer(method=method, n_components=n_components)
    X_red: np.ndarray = reducer.fit_transform(phi_w)

    engine: ClusterEngine = ClusterEngine(method=cluster_method, n_clusters=n_clusters)
    labels: np.ndarray = engine.fit_predict(X_red)

    evaluator: Evaluator = Evaluator()
    metrics: dict[str, Any] = evaluator.evaluate(X_red, labels)

    return {
        "labels": labels,
        "X_reduced": X_red,
        "metrics": metrics,
        "params": {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "n_components": n_components,
            "n_clusters": n_clusters,
            "method": method,
            "cluster_method": cluster_method,
        },
    }


def tune_params(
    e_text: np.ndarray,
    e_type: np.ndarray,
    e_rest: np.ndarray,
    e_stat: np.ndarray,
    skip_bert: bool,
    method: str = "pca",
) -> dict[str, Any]:
    """Grid search over weights (alpha, beta, gamma, delta), n_components and n_clusters.

    Args:
        e_text: BERT/synthetic embeddings, shape (N, 768).
        e_type: One-hot encoded data types.
        e_rest: Binary constraint encodings.
        e_stat: Statistical features.
        skip_bert: Whether BERT was skipped.
        method: Dimensionality reduction method (pca, umap, svd, tsne).

    Returns:
        Best result dict with keys: silhouette, alpha, beta, gamma, delta,
        n_components, n_clusters, labels, X_reduced, metrics.
    """
    logger.info("\n--- AUTOMATIC HYPERPARAMETER TUNING ---")

    alphas: list[float] = [0.25, 0.30, 0.35]
    betas: list[float] = [0.15, 0.20, 0.25, 0.30]
    deltas: list[float] = [0.00, 0.03, 0.05, 0.10, 0.15]

    n_samples: int = e_text.shape[0]
    n_components_opts: list[int] = [min(5, n_samples), min(10, n_samples)]
    n_clusters_opts: list[int] = [3, 5, 7]

    best: dict[str, Any] = {"silhouette": -1}
    weight_combos: int = len(alphas) * len(betas) * len(deltas)
    total: int = weight_combos * len(n_components_opts) * len(n_clusters_opts)
    count: int = 0

    for alpha in alphas:
        for beta in betas:
            for delta in deltas:
                gamma: float = round(1.0 - alpha - beta - delta, 2)
                if gamma < 0.10 or gamma > 0.50:
                    continue
                for nc in n_components_opts:
                    if nc >= n_samples:
                        continue
                    # Build phi and reduce ONCE per (weights, nc)
                    builder_w: FeatureBuilder = FeatureBuilder(alpha=alpha, beta=beta, gamma=gamma, delta=delta)
                    phi_w: np.ndarray = builder_w.build(e_text, e_type, e_rest, e_stat)
                    reducer_w: DimensionalityReducer = DimensionalityReducer(method=method, n_components=nc)
                    X_red_w: np.ndarray = reducer_w.fit_transform(phi_w)
                    for nk in n_clusters_opts:
                        if nk >= n_samples:
                            continue
                        count += 1
                        engine_w: ClusterEngine = ClusterEngine(method="kmeans", n_clusters=nk)
                        labels_w: np.ndarray = engine_w.fit_predict(X_red_w)
                        evaluator_w: Evaluator = Evaluator()
                        metrics_w: dict[str, Any] = evaluator_w.evaluate(X_red_w, labels_w)
                        sil: float = metrics_w["silhouette"]
                        logger.debug(
                            "  combo %d/%d: alpha=%.2f beta=%.2f gamma=%.2f delta=%.2f "
                            "nc=%d nk=%d sil=%.4f",
                            count,
                            total,
                            alpha,
                            beta,
                            gamma,
                            delta,
                            nc,
                            nk,
                            sil,
                        )
                        if sil > best["silhouette"]:
                            best = {
                                "silhouette": sil,
                                "alpha": alpha,
                                "beta": beta,
                                "gamma": gamma,
                                "delta": delta,
                                "n_components": nc,
                                "n_clusters": nk,
                                "labels": labels_w,
                                "X_reduced": X_red_w,
                                "metrics": metrics_w,
                            }

    logger.info(
        "Best combo: alpha=%.2f beta=%.2f gamma=%.2f delta=%.2f  nc=%d nk=%d  silhouette=%.4f",
        best["alpha"],
        best["beta"],
        best["gamma"],
        best["delta"],
        best["n_components"],
        best["n_clusters"],
        best["silhouette"],
    )
    return best


def main() -> None:
    """Main entry point: run the full Phase 2 pipeline from Oracle to visualization."""
    args: argparse.Namespace = parse_args()

    run_tag: str = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.skip_bert:
        run_tag += "_nobert"
    else:
        run_tag += "_bert"
    output_dir: str = args.output_dir or str(Path("../output") / run_tag)

    logger.info("=" * 50)
    logger.info("PHASE 2: ML PIPELINE ON DB SCHEMAS")
    logger.info("=" * 50)
    logger.info("Output dir: %s", output_dir)

    db: OracleConnector = OracleConnector()
    connection = db.connect()

    try:
        # 2. Schema extraction
        logger.info("\n[2/10] Extracting database schema...")
        extractor: SchemaExtractor = SchemaExtractor(connection)
        schema: DatabaseSchema = extractor.extract_all()
        logger.info("  Tables: %d, Columns: %d", len(schema.tables), schema.total_columns())

        all_columns: list[ColumnMetadata] = []
        column_table_map: list[str] = []
        column_names: list[str] = []
        for table in schema.tables:
            for col in table.columns:
                all_columns.append(col)
                column_table_map.append(table.name)
                column_names.append(f"{table.name}.{col.name}")

        n: int = len(all_columns)
        if n == 0:
            logger.error("No columns found. Exiting.")
            return

        # 3. Text preprocessing
        logger.info("\n[3/10] Preprocessing column names...")
        preprocessor: TextPreprocessor = TextPreprocessor()
        processed_texts: list[str] = []
        for col, tname in zip(all_columns, column_table_map):
            text: str = preprocessor.process(col.name, table_name=tname)
            processed_texts.append(text)

        for i in range(min(3, len(processed_texts))):
            logger.info("  %s -> %s", column_names[i], processed_texts[i])

        # 4. BERT embeddings (or synthetic)
        logger.info("\n[4/10] Generating embeddings...")
        embedder: BERTEmbedder | None = None
        e_text: np.ndarray
        if args.skip_bert:
            logger.info("  Using synthetic embeddings (--skip-bert)")
            e_text = np.random.randn(n, 768).astype(np.float32)
        else:
            embedder = BERTEmbedder()
            try:
                e_text = embedder.encode(processed_texts)
                logger.info("  BERT embeddings: %s", e_text.shape)
            except (ImportError, EmbeddingError) as e:
                logger.warning("  Error loading BERT: %s", e)
                logger.info("  Fallback: synthetic embeddings")
                e_text = np.random.randn(n, 768).astype(np.float32)

        # 5. Structural encoding
        logger.info("\n[5/10] Encoding structure...")
        encoder: StructuralEncoder = StructuralEncoder()
        e_type: np.ndarray = encoder.encode_data_types(all_columns)
        e_rest: np.ndarray = encoder.encode_constraints(all_columns)
        e_stat: np.ndarray = encoder.encode_statistical(all_columns)
        logger.info(
            "  Types: %s, Constraints: %s, Statistical: %s",
            e_type.shape,
            e_rest.shape,
            e_stat.shape,
        )

        # 6. Feature vectors (base, with default weights)
        logger.info("\n[6/10] Building composite phi vectors...")
        builder: FeatureBuilder = FeatureBuilder(alpha=0.40, beta=0.30, gamma=0.25, delta=0.05)
        phi: np.ndarray = builder.build(e_text, e_type, e_rest, e_stat)
        logger.info("  Phi vector: %s (dim=%d)", phi.shape, phi.shape[1])

        # ── TUNING ─────────────────────────────────────────────────────
        if args.tune:
            logger.info("\n[7-8] Automatic hyperparameter tuning...")
            best: dict[str, Any] = tune_params(e_text, e_type, e_rest, e_stat, args.skip_bert, method=args.method)
            labels: np.ndarray = best["labels"]
            X_reduced: np.ndarray = best["X_reduced"]
            metrics: dict[str, Any] = best["metrics"]
            alpha: float = best["alpha"]
            beta: float = best["beta"]
            gamma: float = best["gamma"]
            delta: float = best.get("delta", 0.05)
            n_components: int = best["n_components"]
            n_clusters: int = best["n_clusters"]
            method: str = "pca"
        else:
            alpha = args.alpha if args.alpha is not None else 0.40
            beta = args.beta if args.beta is not None else 0.30
            gamma = args.gamma if args.gamma is not None else 0.25
            delta = args.delta if args.delta is not None else 0.05
            # Rebuild phi with user-specified weights if any differ from default
            if (
                args.alpha is not None
                or args.beta is not None
                or args.gamma is not None
                or args.delta is not None
            ):
                builder_custom: FeatureBuilder = FeatureBuilder(
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    delta=delta,
                )
                phi = builder_custom.build(e_text, e_type, e_rest, e_stat)
                logger.info(
                    "  Phi rebuilt with weights alpha=%.2f beta=%.2f gamma=%.2f delta=%.2f: %s",
                    alpha,
                    beta,
                    gamma,
                    delta,
                    phi.shape,
                )

            n_components = args.n_components or min(20, n)
            n_clusters = args.n_clusters or min(5, n)
            method = args.method

            logger.info(
                "\n[7/10] Reducing dimensionality (%s, dims=%d)...",
                method,
                n_components,
            )
            reducer: DimensionalityReducer = DimensionalityReducer(
                method=method,
                n_components=n_components,
            )
            X_reduced = reducer.fit_transform(phi)
            logger.info("  Reduced: %s", X_reduced.shape)

            if reducer.explained_variance_ratio is not None:
                var_exp: float = reducer.explained_variance_ratio.sum()
                logger.info("  Explained variance: %.3f", var_exp)

            logger.info(
                "\n[8/10] Running clustering (%s, clusters=%d)...",
                args.cluster_method,
                n_clusters,
            )
            dbscan_kwargs: dict[str, Any] = {}
            if args.cluster_method == "dbscan":
                if args.dbscan_eps is not None:
                    eps_val: str | float = args.dbscan_eps
                    if eps_val != "auto":
                        eps_val = float(eps_val)
                    dbscan_kwargs["eps"] = eps_val
                if args.dbscan_min_pts is not None:
                    dbscan_kwargs["min_samples"] = args.dbscan_min_pts
                dbscan_kwargs["metric"] = args.metric
            elif args.cluster_method == "hdbscan":
                dbscan_kwargs["min_cluster_size"] = args.hdbscan_min_cluster_size
                if args.hdbscan_min_samples is not None:
                    dbscan_kwargs["min_samples"] = args.hdbscan_min_samples
                dbscan_kwargs["metric"] = args.metric

            engine: ClusterEngine = ClusterEngine(
                method=args.cluster_method,
                n_clusters=n_clusters,
                **dbscan_kwargs,
            )
            labels = engine.fit_predict(X_reduced)
            unique_labels = sorted(set(labels))
            n_noise: int = int((labels == -1).sum())
            logger.info(
                "  Clusters: %d (%d noise)",
                len(unique_labels) - (1 if -1 in unique_labels else 0),
                n_noise,
            )

            for label in unique_labels:
                count: int = int((labels == label).sum())
                logger.info("    Cluster %d: %d columns", label, count)

            metrics = Evaluator().evaluate(X_reduced, labels)

        # 9. Detailed evaluation
        logger.info("\n[9/10] Evaluating clustering quality...")
        evaluator = Evaluator()
        cross_table: dict[str, Any] = evaluator.cross_table_analysis(column_table_map, labels)
        composition: dict[int, dict[str, Any]] = evaluator.cluster_composition(labels, all_columns)

        if not args.tune:
            metrics = evaluator.evaluate(X_reduced, labels)

        report: str = evaluator.print_report(metrics, cross_table, composition)
        report_path: Path = Path(output_dir) / "evaluation_report.txt"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report)
        logger.info("\n%s", report)

        # 10. Recommendations
        logger.info("\n[10/10] Generating recommendations...")
        recommender: Recommender = Recommender()
        recommendations: list[dict[str, Any]] = recommender.recommend(
            schema, labels, column_table_map
        )
        rec_text: str = recommender.print_recommendations(recommendations)
        rec_path: Path = Path(output_dir) / "recommendations.txt"
        with open(rec_path, "w") as f:
            f.write(rec_text)
        logger.info("\n%s", rec_text)

        # 11. Ground truth validation (H1)
        if args.validate:
            logger.info("\n[11/11] Validating against anti-pattern ground truth...")
            gt_map: dict[str, str] = build_ground_truth_map()
            labels_true: list[str | None] = get_ground_truth(
                column_table_map,
                column_names,
                gt_map,
            )
            validation: dict[str, Any] = evaluator.validate_against_ground_truth(
                labels,
                labels_true,
                X=X_reduced,
            )
            val_report: str = Evaluator.print_validation_report(validation)
            val_path: Path = Path(output_dir) / "validation_report.txt"
            with open(val_path, "w") as f:
                f.write(val_report)
            logger.info("\n%s", val_report)

        # 12. Visualization
        viz: Visualizer = Visualizer(output_dir=output_dir)
        viz_files: dict[str, str] = viz.save_all(
            X_reduced,
            labels,
            column_table_map,
            all_columns,
            column_names,
        )
        for name, path in viz_files.items():
            if path:
                logger.info("  %s: %s", name, path)

        logger.info("\n" + "=" * 50)
        logger.info("PHASE 2 COMPLETED")
        logger.info(
            "Parameters: alpha=%.2f beta=%.2f gamma=%.2f delta=%.2f | dims=%d | clusters=%d",
            alpha,
            beta,
            gamma,
            delta,
            n_components,
            n_clusters,
        )
        logger.info("Report: %s", report_path)
        logger.info("Recommendations: %s", rec_path)
        logger.info("Charts: %s/", output_dir)
        logger.info("=" * 50)

    except (BadDBError, DatabaseError, EmbeddingError) as e:
        logger.error("Pipeline error (Phase 2): %s", e, exc_info=True)
    except Exception as e:
        logger.error("Unexpected error in pipeline (Phase 2): %s", e, exc_info=True)
    finally:
        db.close()
        if embedder is not None:
            try:
                embedder.unload()
            except BadDBError:
                logger.debug("Error unloading BERT embedder (ignored)")


if __name__ == "__main__":
    main()
