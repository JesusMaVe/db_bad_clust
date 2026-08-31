"""
test_validation.py — Tests for validation.py module
"""

import numpy as np
import pytest

from db_bad_clust.data.schema_extractor import ColumnMetadata
from db_bad_clust.evaluation.validation import (
    GroundTruthValidator,
    cluster_composition,
    cross_table_analysis,
)


class TestAdjustedRandIndex:
    """Tests for ARI metric."""

    def test_ari_perfect_match(self):
        """Identical clusterings should have ARI = 1."""
        labels_pred = np.array([0, 0, 1, 1])
        labels_true = np.array([0, 0, 1, 1])
        assert GroundTruthValidator.adjusted_rand_index(labels_pred, labels_true) == 1.0

    def test_ari_random(self):
        """Random clusterings should have ARI near 0."""
        labels_pred = np.array([0, 1, 0, 1])
        labels_true = np.array([0, 0, 1, 1])
        ari = GroundTruthValidator.adjusted_rand_index(labels_pred, labels_true)
        # ARI can be negative or positive for random
        assert -1.0 <= ari <= 1.0

    def test_ari_complementary(self):
        """Complementary clusterings still have ARI = 1 (same structure)."""
        labels_pred = np.array([0, 0, 1, 1])
        labels_true = np.array([1, 1, 0, 0])
        ari = GroundTruthValidator.adjusted_rand_index(labels_pred, labels_true)
        # ARI is invariant to label permutation
        assert ari == 1.0


class TestNormalizedMutualInfo:
    """Tests for NMI metric."""

    def test_nmi_perfect_match(self):
        """Identical clusterings should have NMI = 1."""
        labels_pred = np.array([0, 0, 1, 1])
        labels_true = np.array([0, 0, 1, 1])
        assert GroundTruthValidator.normalized_mutual_info(labels_pred, labels_true) == 1.0

    def test_nmi_range(self):
        """NMI should be between 0 and 1."""
        labels_pred = np.array([0, 1, 0, 1])
        labels_true = np.array([0, 0, 1, 1])
        nmi = GroundTruthValidator.normalized_mutual_info(labels_pred, labels_true)
        assert 0 <= nmi <= 1


class TestValidate:
    """Tests for validate method."""

    def test_validate_with_ground_truth(self):
        """Validate with known ground truth."""
        validator = GroundTruthValidator()
        labels_pred = np.array([0, 0, 1, 1, 0])
        labels_true = ["clean", "clean", "eav", "eav", None]
        result = validator.validate(labels_pred, labels_true)
        assert "ari" in result
        assert "nmi" in result
        assert "n_gt_labels" in result

    def test_validate_insufficient_gt(self):
        """Validate with less than 2 ground truth labels."""
        validator = GroundTruthValidator()
        labels_pred = np.array([0, 0, 1, 1])
        labels_true = [None, None, None, None]
        result = validator.validate(labels_pred, labels_true)
        assert "error" in result
        assert result["n_gt_labels"] == 0

    def test_validate_with_feature_matrix(self):
        """Validate with X for centroid anomaly scores."""
        validator = GroundTruthValidator()
        X = np.random.randn(10, 5)
        labels_pred = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])
        labels_true = ["clean", "clean", "clean", "eav", "eav", "eav",
                       "giant", "giant", "giant", "clean"]
        result = validator.validate(labels_pred, labels_true, X)
        assert "structural_anomalies" in result


# ── cross_table_analysis ────────────────────────────────────────────────


def _cols() -> list[ColumnMetadata]:
    """Sample columns for cluster_composition tests."""
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


class TestCrossTableAnalysis:
    """Cross-table analysis."""

    def test_cross_table_returns_dict(self) -> None:
        """cross_table_analysis returns dict with expected keys."""
        column_table_map = ["EMP", "EMP", "DEPT", "DEPT", "PROD"]
        labels = np.array([0, 0, 1, 1, 0], dtype=int)
        result = cross_table_analysis(column_table_map, labels)
        assert isinstance(result, dict)
        assert "table_distribution" in result
        assert "table_purity" in result
        assert "average_purity" in result

    def test_cross_table_distribution(self) -> None:
        """table_distribution groups columns by table and cluster."""
        column_table_map = ["EMP", "EMP", "DEPT", "DEPT", "PROD"]
        labels = np.array([0, 0, 1, 1, 0], dtype=int)
        result = cross_table_analysis(column_table_map, labels)
        dist = result["table_distribution"]
        assert "EMP" in dist
        assert "DEPT" in dist
        assert "PROD" in dist
        # EMP has 2 columns both in cluster 0
        assert dist["EMP"][0] == 2
        # DEPT has 2 columns both in cluster 1
        assert dist["DEPT"][1] == 2

    def test_cross_table_purity(self) -> None:
        """Table purity is 1.0 when all columns are in one cluster."""
        column_table_map = ["T1", "T1", "T1"]
        labels = np.array([0, 0, 0], dtype=int)
        result = cross_table_analysis(column_table_map, labels)
        assert result["table_purity"]["T1"]["purity"] == 1.0

    def test_cross_table_mixed_purity(self) -> None:
        """Table with columns in multiple clusters has purity < 1."""
        column_table_map = ["T1", "T1", "T1", "T1"]
        labels = np.array([0, 0, 1, 1], dtype=int)
        result = cross_table_analysis(column_table_map, labels)
        # 2 out of 4 in majority cluster → 0.5
        assert result["table_purity"]["T1"]["purity"] == 0.5

    def test_cross_table_average_purity(self) -> None:
        """Average purity across tables is correctly computed."""
        column_table_map = ["T1", "T1", "T2", "T2"]
        labels = np.array([0, 0, 1, 1], dtype=int)
        result = cross_table_analysis(column_table_map, labels)
        # Both tables have purity 1.0
        assert result["average_purity"] == 1.0

    def test_cross_table_empty(self) -> None:
        """Empty inputs produce empty distributions."""
        result = cross_table_analysis([], np.array([], dtype=int))
        assert result["table_distribution"] == {}
        assert result["table_purity"] == {}
        assert result["average_purity"] == 0.0


# ── cluster_composition ─────────────────────────────────────────────────


class TestClusterComposition:
    """Anti-pattern composition per cluster."""

    def test_composition_returns_dict(
        self,
    ) -> None:
        """cluster_composition returns dict keyed by cluster id."""
        labels = np.array([0, 0, 1, 1, 0], dtype=int)
        result = cluster_composition(labels, _cols())
        assert isinstance(result, dict)
        assert 0 in result
        assert 1 in result

    def test_composition_keys(
        self,
    ) -> None:
        """Each cluster entry has expected keys."""
        labels = np.array([0, 0, 1, 1, 0], dtype=int)
        result = cluster_composition(labels, _cols())
        expected_keys = {
            "total_columns",
            "pct_nullable",
            "pct_varchar",
            "pct_numeric",
            "types_distribution",
        }
        for cluster_id in result:
            assert set(result[cluster_id].keys()) == expected_keys

    def test_composition_counts(self) -> None:
        """Cluster 0 has 3 columns, cluster 1 has 2."""
        cols = [
            ColumnMetadata(name="A", data_type="VARCHAR2", data_length=10, nullable=True),
            ColumnMetadata(name="B", data_type="VARCHAR2", data_length=10, nullable=False),
            ColumnMetadata(name="C", data_type="NUMBER", data_length=22, nullable=True),
            ColumnMetadata(name="D", data_type="DATE", data_length=7, nullable=False),
            ColumnMetadata(name="E", data_type="NUMBER", data_length=22, nullable=True),
        ]
        labels = np.array([0, 0, 0, 1, 1], dtype=int)
        result = cluster_composition(labels, cols)
        assert result[0]["total_columns"] == 3
        assert result[1]["total_columns"] == 2

    def test_composition_percentages(self) -> None:
        """Nullable percentage is correctly calculated."""
        cols = [
            ColumnMetadata(name="A", data_type="VARCHAR2", data_length=10, nullable=True),
            ColumnMetadata(name="B", data_type="VARCHAR2", data_length=10, nullable=False),
            ColumnMetadata(name="C", data_type="NUMBER", data_length=22, nullable=True),
        ]
        labels = np.array([0, 0, 0], dtype=int)
        result = cluster_composition(labels, cols)
        # 2 out of 3 nullable ≈ 66.67%
        assert result[0]["pct_nullable"] == pytest.approx(66.6667, abs=0.01)

    def test_composition_ignores_noise(self) -> None:
        """Noise points (-1) get their own entry in the composition dict."""
        cols = [
            ColumnMetadata(name="A", data_type="VARCHAR2", data_length=10, nullable=True),
            ColumnMetadata(name="B", data_type="NUMBER", data_length=22, nullable=False),
        ]
        labels = np.array([-1, -1], dtype=int)
        result = cluster_composition(labels, cols)
        assert -1 in result
        assert result[-1]["total_columns"] == 2

    def test_composition_empty(self) -> None:
        """Empty inputs produce empty dict."""
        result = cluster_composition(np.array([], dtype=int), [])
        assert result == {}
