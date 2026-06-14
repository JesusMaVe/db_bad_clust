"""
profile_performance.py — Performance profiling for db_bad_clust

Usage:
    python profile_performance.py
"""

import cProfile
import pstats
from pstats import SortKey
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def profile_feature_builder():
    """Profile FeatureBuilder operations."""
    from feature_builder import FeatureBuilder

    builder = FeatureBuilder(alpha=0.35, beta=0.15, gamma=0.50, delta=0.00)

    # Simulate data: 235 columns (as in your dataset)
    N = 235
    e_text = np.random.randn(N, 768).astype(np.float32)
    e_type = np.random.randn(N, 12).astype(np.float32)
    e_rest = np.random.randn(N, 5).astype(np.float32)

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(10):
        phi = builder.build(e_text, e_type, e_rest)

    profiler.disable()
    return profiler


def profile_clustering():
    """Profile ClusterEngine operations."""
    from cluster_engine import ClusterEngine

    N = 235
    d = 5
    X = np.random.randn(N, d).astype(np.float32)

    profiler = cProfile.Profile()
    profiler.enable()

    for method in ["kmeans", "dbscan", "agglomerative"]:
        if method == "dbscan":
            engine = ClusterEngine(method=method, eps="auto")
        else:
            engine = ClusterEngine(method=method, n_clusters=5)
        labels = engine.fit_predict(X)

    profiler.disable()
    return profiler


def profile_evaluator():
    """Profile Evaluator operations."""
    from evaluator import Evaluator

    evaluator = Evaluator()
    N = 235
    d = 5
    X = np.random.randn(N, d).astype(np.float32)
    labels = np.random.randint(0, 5, N)

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(10):
        metrics = evaluator.evaluate(X, labels)

    profiler.disable()
    return profiler


def profile_dimensionality_reducer():
    """Profile DimensionalityReducer operations."""
    from dimensionality_reducer import DimensionalityReducer

    N = 235
    d = 785  # 768 + 12 + 5
    X = np.random.randn(N, d).astype(np.float32)

    profiler = cProfile.Profile()
    profiler.enable()

    for method in ["pca", "umap"]:
        reducer = DimensionalityReducer(method=method, n_components=5)
        X_reduced = reducer.fit_transform(X)

    profiler.disable()
    return profiler


def profile_text_preprocessor():
    """Profile TextPreprocessor operations."""
    from text_preprocessor import TextPreprocessor

    preprocessor = TextPreprocessor()

    # Simulate column names
    column_names = [
        "EMP_ID", "EMP_NAME", "EMP_SALARY", "DEPT_ID", "DEPT_NAME",
        "HIRE_DATE", "JOB_TITLE", "MANAGER_ID", "LOCATION", "PHONE",
    ] * 25  # 250 columns

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(10):
        processed = [preprocessor.preprocess(name) for name in column_names]

    profiler.disable()
    return profiler


def print_stats(profiler, title):
    """Print profiling statistics."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(15)


def benchmark_simple_operations():
    """Benchmark simple Python vs NumPy operations."""
    print(f"\n{'='*60}")
    print("  BENCHMARK: Python vs NumPy")
    print(f"{'='*60}")

    N = 100000

    # Z-score: Python vs NumPy
    data = np.random.randn(N, 768).astype(np.float32)

    # NumPy (current implementation)
    start = time.perf_counter()
    for _ in range(10):
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        result = (data - mean) / std
    numpy_time = time.perf_counter() - start
    print(f"  NumPy Z-score (10 runs): {numpy_time:.4f}s")

    # Vectorized distance computation
    N_small = 235
    X = np.random.randn(N_small, 5).astype(np.float32)

    # Loop-based (current in centroid_anomaly_scores)
    start = time.perf_counter()
    for _ in range(100):
        centroids = {}
        unique_labels = np.unique(np.random.randint(0, 5, N_small))
        for cid in unique_labels:
            mask = np.random.randint(0, 5, N_small) == cid
            centroids[cid] = X[mask].mean(axis=0)
        distances = np.array([
            float(np.linalg.norm(X[i] - centroids[i % 5]))
            for i in range(N_small)
        ])
    loop_time = time.perf_counter() - start
    print(f"  Loop-based distances (100 runs): {loop_time:.4f}s")

    # Vectorized
    start = time.perf_counter()
    for _ in range(100):
        labels_rand = np.random.randint(0, 5, N_small)
        centroids = {}
        for cid in np.unique(labels_rand):
            mask = labels_rand == cid
            centroids[cid] = X[mask].mean(axis=0)
        centroid_array = np.array([centroids[int(l)] for l in labels_rand])
        distances = np.linalg.norm(X - centroid_array, axis=1)
    vectorized_time = time.perf_counter() - start
    print(f"  Vectorized distances (100 runs): {vectorized_time:.4f}s")
    print(f"  Speedup: {loop_time/vectorized_time:.2f}x")


if __name__ == "__main__":
    print("Python Performance Profiling for db_bad_clust")
    print("=" * 60)

    # Run benchmarks
    benchmark_simple_operations()

    # Profile different modules
    profilers = {
        "FeatureBuilder (build)": profile_feature_builder,
        "ClusterEngine (all methods)": profile_clustering,
        "Evaluator (evaluate)": profile_evaluator,
        "DimensionalityReducer (PCA + UMAP)": profile_dimensionality_reducer,
        "TextPreprocessor": profile_text_preprocessor,
    }

    for title, func in profilers.items():
        try:
            profiler = func()
            print_stats(profiler, title)
        except Exception as e:
            print(f"\n  Error profiling {title}: {e}")

    print(f"\n{'='*60}")
    print("  PROFILING COMPLETE")
    print(f"{'='*60}")
