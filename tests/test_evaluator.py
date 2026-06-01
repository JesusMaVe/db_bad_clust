"""
test_evaluator.py — Tests for the Evaluator module.

Verifies:
  - Individual metrics (silhouette, davies_bouldin, calinski_harabasz) return floats
  - Edge cases: 0/1 samples, single cluster, all noise
  - evaluate() returns dict with expected keys
  - cross_table_analysis returns expected structure
  - cluster_composition returns expected structure
  - print_report returns a formatted string
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest
import numpy as np
from evaluator import Evaluator
from schema_extractor import ColumnMetadata


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def evaluator() -> Evaluator:
    """Default Evaluator instance."""
    return Evaluator()


@pytest.fixture
def well_separated_data() -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated clusters (10 samples, 2D) with known labels."""
    np.random.seed(42)
    c0 = np.random.randn(5, 2) + np.array([5, 5])
    c1 = np.random.randn(5, 2) + np.array([-5, -5])
    X = np.vstack([c0, c1]).astype(np.float32)
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int)
    return X, labels


@pytest.fixture
def noisy_data() -> tuple[np.ndarray, np.ndarray]:
    """Data where some points are outliers (-1 labels)."""
    np.random.seed(7)
    X = np.random.randn(10, 2).astype(np.float32)
    labels = np.array([0, 0, 0, 1, 1, -1, -1, 0, 1, -1], dtype=int)
    return X, labels


@pytest.fixture
def single_cluster_data() -> tuple[np.ndarray, np.ndarray]:
    """All points in one cluster."""
    X = np.random.randn(10, 2).astype(np.float32)
    labels = np.zeros(10, dtype=int)
    return X, labels


@pytest.fixture
def few_points_data() -> tuple[np.ndarray, np.ndarray]:
    """Single point (edge case for metrics)."""
    X = np.random.randn(1, 2).astype(np.float32)
    labels = np.array([0], dtype=int)
    return X, labels


@pytest.fixture
def column_metadata_for_composition() -> list[ColumnMetadata]:
    """Column metadata for cluster_composition tests."""
    return [
        ColumnMetadata(
            name="ID", data_type="NUMBER", data_length=22, nullable=False, is_primary_key=True
        ),
        ColumnMetadata(name="NOMBRE", data_type="VARCHAR2", data_length=100, nullable=True),
        ColumnMetadata(name="SALARIO", data_type="NUMBER", data_length=22, nullable=True),
        ColumnMetadata(name="ACTIVO", data_type="CHAR", data_length=1, nullable=False),
        ColumnMetadata(
            name="EMAIL", data_type="VARCHAR2", data_length=200, nullable=True, is_unique=True
        ),
    ]


# ── Individual metrics ──────────────────────────────────────────────────


class TestIndividualMetrics:
    """Each metric returns a float."""

    def test_silhouette(
        self, evaluator: Evaluator, well_separated_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """silhouette returns a float between -1 and 1."""
        X, labels = well_separated_data
        score = evaluator.silhouette(X, labels)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0
        # Well-separated clusters should have positive score
        assert score > 0.5

    def test_davies_bouldin(
        self, evaluator: Evaluator, well_separated_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """davies_bouldin returns a non-negative float."""
        X, labels = well_separated_data
        score = evaluator.davies_bouldin(X, labels)
        assert isinstance(score, float)
        assert score >= 0.0
        # Well-separated clusters → low D-B
        assert score < 1.0

    def test_calinski_harabasz(
        self, evaluator: Evaluator, well_separated_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """calinski_harabasz returns a positive float."""
        X, labels = well_separated_data
        score = evaluator.calinski_harabasz(X, labels)
        assert isinstance(score, (float, np.floating))
        assert score > 0.0


# ── Edge cases for individual metrics ────────────────────────────────────


class TestMetricsEdgeCases:
    """Metrics handle degenerate cases gracefully."""

    def test_silhouette_single_cluster(
        self, evaluator: Evaluator, single_cluster_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Single cluster → silhouette returns 0.0."""
        X, labels = single_cluster_data
        assert evaluator.silhouette(X, labels) == 0.0

    def test_silhouette_one_point(
        self, evaluator: Evaluator, few_points_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Single point → silhouette returns 0.0."""
        X, labels = few_points_data
        assert evaluator.silhouette(X, labels) == 0.0

    def test_silhouette_all_noise(self, evaluator: Evaluator) -> None:
        """All labels are -1 (noise) → returns 0.0."""
        X = np.random.randn(10, 2).astype(np.float32)
        labels = np.full(10, -1, dtype=int)
        assert evaluator.silhouette(X, labels) == 0.0

    def test_davies_bouldin_single_cluster(
        self, evaluator: Evaluator, single_cluster_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Single cluster → davies_bouldin returns 0.0."""
        X, labels = single_cluster_data
        assert evaluator.davies_bouldin(X, labels) == 0.0

    def test_davies_bouldin_all_noise(self, evaluator: Evaluator) -> None:
        """All labels -1 → davies_bouldin returns 0.0."""
        X = np.random.randn(10, 2).astype(np.float32)
        labels = np.full(10, -1, dtype=int)
        assert evaluator.davies_bouldin(X, labels) == 0.0

    def test_calinski_harabasz_single_cluster(
        self, evaluator: Evaluator, single_cluster_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Single cluster → calinski_harabasz returns 0.0."""
        X, labels = single_cluster_data
        assert evaluator.calinski_harabasz(X, labels) == 0.0

    def test_calinski_harabasz_all_noise(self, evaluator: Evaluator) -> None:
        """All labels -1 → calinski_harabasz returns 0.0."""
        X = np.random.randn(10, 2).astype(np.float32)
        labels = np.full(10, -1, dtype=int)
        assert evaluator.calinski_harabasz(X, labels) == 0.0


# ── evaluate() ───────────────────────────────────────────────────────────


class TestEvaluate:
    """Full evaluate() method."""

    def test_evaluate_returns_dict(
        self, evaluator: Evaluator, well_separated_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """evaluate() returns a dict."""
        X, labels = well_separated_data
        result = evaluator.evaluate(X, labels)
        assert isinstance(result, dict)

    def test_evaluate_expected_keys(
        self, evaluator: Evaluator, well_separated_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """evaluate() dict has expected keys."""
        X, labels = well_separated_data
        result = evaluator.evaluate(X, labels)
        expected_keys = {
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
            "n_clusters",
            "n_samples",
            "n_noise",
        }
        assert set(result.keys()) == expected_keys

    def test_evaluate_n_clusters(
        self, evaluator: Evaluator, well_separated_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """n_clusters matches input."""
        X, labels = well_separated_data
        result = evaluator.evaluate(X, labels)
        assert result["n_clusters"] == 2

    def test_evaluate_n_samples(
        self, evaluator: Evaluator, well_separated_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """n_samples matches input."""
        X, labels = well_separated_data
        result = evaluator.evaluate(X, labels)
        assert result["n_samples"] == 10

    def test_evaluate_noise(
        self, evaluator: Evaluator, noisy_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Noise points are counted in n_noise."""
        X, labels = noisy_data
        result = evaluator.evaluate(X, labels)
        assert result["n_noise"] == 3  # 3 points labeled -1
        assert result["n_samples"] == 10

    def test_evaluate_all_noise(self, evaluator: Evaluator) -> None:
        """All noise → returns error dict."""
        X = np.random.randn(10, 2).astype(np.float32)
        labels = np.full(10, -1, dtype=int)
        result = evaluator.evaluate(X, labels)
        assert "error" in result
        assert result["error"] == "No samples with assigned cluster"

    def test_evaluate_metrics_are_floats(
        self, evaluator: Evaluator, well_separated_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Individual metric values are floats (not numpy floats)."""
        X, labels = well_separated_data
        result = evaluator.evaluate(X, labels)
        assert isinstance(result["silhouette"], float)
        assert isinstance(result["davies_bouldin"], float)
        assert isinstance(result["calinski_harabasz"], (float, np.floating))


# ── cross_table_analysis ────────────────────────────────────────────────


class TestCrossTableAnalysis:
    """Cross-table analysis."""

    def test_cross_table_returns_dict(self, evaluator: Evaluator) -> None:
        """cross_table_analysis returns dict with expected keys."""
        column_table_map = ["EMP", "EMP", "DEPT", "DEPT", "PROD"]
        labels = np.array([0, 0, 1, 1, 0], dtype=int)
        result = evaluator.cross_table_analysis(column_table_map, labels)
        assert isinstance(result, dict)
        assert "table_distribution" in result
        assert "table_purity" in result
        assert "average_purity" in result

    def test_cross_table_distribution(self, evaluator: Evaluator) -> None:
        """table_distribution groups columns by table and cluster."""
        column_table_map = ["EMP", "EMP", "DEPT", "DEPT", "PROD"]
        labels = np.array([0, 0, 1, 1, 0], dtype=int)
        result = evaluator.cross_table_analysis(column_table_map, labels)
        dist = result["table_distribution"]
        assert "EMP" in dist
        assert "DEPT" in dist
        assert "PROD" in dist
        # EMP has 2 columns both in cluster 0
        assert dist["EMP"][0] == 2
        # DEPT has 2 columns both in cluster 1
        assert dist["DEPT"][1] == 2

    def test_cross_table_purity(self, evaluator: Evaluator) -> None:
        """Table purity is 1.0 when all columns are in one cluster."""
        column_table_map = ["T1", "T1", "T1"]
        labels = np.array([0, 0, 0], dtype=int)
        result = evaluator.cross_table_analysis(column_table_map, labels)
        assert result["table_purity"]["T1"]["purity"] == 1.0

    def test_cross_table_mixed_purity(self, evaluator: Evaluator) -> None:
        """Table with columns in multiple clusters has purity < 1."""
        column_table_map = ["T1", "T1", "T1", "T1"]
        labels = np.array([0, 0, 1, 1], dtype=int)
        result = evaluator.cross_table_analysis(column_table_map, labels)
        # 2 out of 4 in majority cluster → 0.5
        assert result["table_purity"]["T1"]["purity"] == 0.5

    def test_cross_table_average_purity(self, evaluator: Evaluator) -> None:
        """Average purity across tables is correctly computed."""
        column_table_map = ["T1", "T1", "T2", "T2"]
        labels = np.array([0, 0, 1, 1], dtype=int)
        result = evaluator.cross_table_analysis(column_table_map, labels)
        # Both tables have purity 1.0
        assert result["average_purity"] == 1.0

    def test_cross_table_empty(self, evaluator: Evaluator) -> None:
        """Empty inputs produce empty distributions."""
        result = evaluator.cross_table_analysis([], np.array([], dtype=int))
        assert result["table_distribution"] == {}
        assert result["table_purity"] == {}
        assert result["average_purity"] == 0.0


# ── cluster_composition ─────────────────────────────────────────────────


class TestClusterComposition:
    """Anti-pattern composition per cluster."""

    def test_composition_returns_dict(
        self, evaluator: Evaluator, column_metadata_for_composition: list[ColumnMetadata]
    ) -> None:
        """cluster_composition returns dict keyed by cluster id."""
        labels = np.array([0, 0, 1, 1, 0], dtype=int)
        result = evaluator.cluster_composition(labels, column_metadata_for_composition)
        assert isinstance(result, dict)
        assert 0 in result
        assert 1 in result

    def test_composition_keys(
        self, evaluator: Evaluator, column_metadata_for_composition: list[ColumnMetadata]
    ) -> None:
        """Each cluster entry has expected keys."""
        labels = np.array([0, 0, 1, 1, 0], dtype=int)
        result = evaluator.cluster_composition(labels, column_metadata_for_composition)
        expected_keys = {
            "total_columns",
            "pct_nullable",
            "pct_varchar",
            "pct_numeric",
            "types_distribution",
        }
        for cluster_id in result:
            assert set(result[cluster_id].keys()) == expected_keys

    def test_composition_counts(self, evaluator: Evaluator) -> None:
        """Cluster 0 has 3 columns, cluster 1 has 2."""
        cols = [
            ColumnMetadata(name="A", data_type="VARCHAR2", data_length=10, nullable=True),
            ColumnMetadata(name="B", data_type="VARCHAR2", data_length=10, nullable=False),
            ColumnMetadata(name="C", data_type="NUMBER", data_length=22, nullable=True),
            ColumnMetadata(name="D", data_type="DATE", data_length=7, nullable=False),
            ColumnMetadata(name="E", data_type="NUMBER", data_length=22, nullable=True),
        ]
        labels = np.array([0, 0, 0, 1, 1], dtype=int)
        result = evaluator.cluster_composition(labels, cols)
        assert result[0]["total_columns"] == 3
        assert result[1]["total_columns"] == 2

    def test_composition_percentages(self, evaluator: Evaluator) -> None:
        """Nullable percentage is correctly calculated."""
        cols = [
            ColumnMetadata(name="A", data_type="VARCHAR2", data_length=10, nullable=True),
            ColumnMetadata(name="B", data_type="VARCHAR2", data_length=10, nullable=False),
            ColumnMetadata(name="C", data_type="NUMBER", data_length=22, nullable=True),
        ]
        labels = np.array([0, 0, 0], dtype=int)
        result = evaluator.cluster_composition(labels, cols)
        # 2 out of 3 nullable ≈ 66.67%
        assert result[0]["pct_nullable"] == pytest.approx(66.6667, abs=0.01)

    def test_composition_ignores_noise(self, evaluator: Evaluator) -> None:
        """Noise points (-1) get their own entry in the composition dict."""
        cols = [
            ColumnMetadata(name="A", data_type="VARCHAR2", data_length=10, nullable=True),
            ColumnMetadata(name="B", data_type="NUMBER", data_length=22, nullable=False),
        ]
        labels = np.array([-1, -1], dtype=int)
        result = evaluator.cluster_composition(labels, cols)
        assert -1 in result
        assert result[-1]["total_columns"] == 2

    def test_composition_empty(self, evaluator: Evaluator) -> None:
        """Empty inputs produce empty dict."""
        result = evaluator.cluster_composition(np.array([], dtype=int), [])
        assert result == {}


# ── print_report ────────────────────────────────────────────────────────


class TestPrintReport:
    """Formatted report generation."""

    def test_print_report_returns_string(self, evaluator: Evaluator) -> None:
        """print_report returns a multi-line string."""
        metrics = {
            "silhouette": 0.75,
            "davies_bouldin": 0.35,
            "calinski_harabasz": 150.0,
            "n_clusters": 3,
            "n_samples": 100,
            "n_noise": 0,
        }
        report = evaluator.print_report(metrics)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_print_report_contains_metrics(self, evaluator: Evaluator) -> None:
        """Report includes metric values."""
        metrics = {
            "silhouette": 0.75,
            "davies_bouldin": 0.35,
            "calinski_harabasz": 150.0,
            "n_clusters": 3,
            "n_samples": 100,
            "n_noise": 0,
        }
        report = evaluator.print_report(metrics)
        assert "0.7500" in report
        assert "0.3500" in report
        assert "150.00" in report
        assert "3" in report
        assert "100" in report

    def test_print_report_with_cross_table(self, evaluator: Evaluator) -> None:
        """Report includes cross-table information."""
        metrics = {
            "silhouette": 0.6,
            "davies_bouldin": 0.4,
            "calinski_harabasz": 100.0,
            "n_clusters": 2,
            "n_samples": 10,
            "n_noise": 0,
        }
        cross_table = {
            "average_purity": 0.85,
            "table_purity": {
                "EMP": {
                    "purity": 0.8,
                    "total_columns": 5,
                    "majority_cluster": 0,
                    "majority_count": 4,
                    "cluster_distribution": {0: 4, 1: 1},
                },
            },
            "table_distribution": {},
        }
        report = evaluator.print_report(metrics, cross_table=cross_table)
        assert "85.00%" in report or "85%" in report
        assert "EMP" in report

    def test_print_report_with_composition(self, evaluator: Evaluator) -> None:
        """Report includes cluster composition."""
        metrics = {
            "silhouette": 0.6,
            "davies_bouldin": 0.4,
            "calinski_harabasz": 100.0,
            "n_clusters": 2,
            "n_samples": 10,
            "n_noise": 0,
        }
        composition = {
            0: {
                "total_columns": 5,
                "pct_nullable": 60.0,
                "pct_varchar": 40.0,
                "pct_numeric": 20.0,
                "types_distribution": {"VARCHAR2": 2, "NUMBER": 1},
            },
            1: {
                "total_columns": 5,
                "pct_nullable": 20.0,
                "pct_varchar": 60.0,
                "pct_numeric": 40.0,
                "types_distribution": {"VARCHAR2": 3, "NUMBER": 2},
            },
        }
        report = evaluator.print_report(metrics, composition=composition)
        assert "Cluster #0" in report
        assert "60%" in report
        assert "40%" in report

    def test_print_report_error(self, evaluator: Evaluator) -> None:
        """Error metrics produce error report."""
        metrics = {"error": "No samples with assigned cluster"}
        report = evaluator.print_report(metrics)
        assert "ERROR" in report
        assert "No samples" in report

    def test_print_report_with_all_args(self, evaluator: Evaluator) -> None:
        """Report with all optional args."""
        metrics = {
            "silhouette": 0.8,
            "davies_bouldin": 0.2,
            "calinski_harabasz": 200.0,
            "n_clusters": 3,
            "n_samples": 30,
            "n_noise": 2,
        }
        cross_table = {"average_purity": 0.9, "table_purity": {}, "table_distribution": {}}
        composition = {
            0: {
                "total_columns": 10,
                "pct_nullable": 50.0,
                "pct_varchar": 30.0,
                "pct_numeric": 70.0,
                "types_distribution": {},
            },
        }
        report = evaluator.print_report(metrics, cross_table=cross_table, composition=composition)
        assert "EVALUACION DE CLUSTERING" in report
        assert "Metricas Internas" in report
        assert "Distribucion por Tabla" in report
        assert "Composicion por Cluster" in report
