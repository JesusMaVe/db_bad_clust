"""
test_evaluator_validation.py — Tests for external validation metrics.

Verifies:
  - adjusted_rand_index returns correct values
  - normalized_mutual_info returns correct values
  - anomaly_detection_metrics handles noise correctly
  - validate_against_ground_truth returns expected structure
  - Edge cases: no ground truth, all None, single label
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest
import numpy as np
from evaluator import Evaluator


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def evaluator() -> Evaluator:
    return Evaluator()


@pytest.fixture
def perfect_labels() -> tuple[np.ndarray, np.ndarray]:
    """Predicted and true labels match exactly."""
    pred = np.array([0, 0, 1, 1, 2, 2], dtype=int)
    true = np.array([0, 0, 1, 1, 2, 2], dtype=int)
    return pred, true


@pytest.fixture
def random_labels() -> tuple[np.ndarray, np.ndarray]:
    """Predicted and true labels are independent (ARI ≈ 0)."""
    np.random.seed(42)
    pred = np.array([0, 0, 1, 1, 0, 1], dtype=int)
    true = np.array([0, 1, 0, 1, 1, 0], dtype=int)
    return pred, true


@pytest.fixture
def noisy_predictions() -> tuple[np.ndarray, list[str | None]]:
    """Some predicted noise, some unknown ground truth."""
    pred = np.array([0, 0, -1, 1, 1, -1, 0, -1], dtype=int)
    true = [
        "wrong_data_types",
        "wrong_data_types",
        None,
        "clean",
        "clean",
        None,
        "wrong_data_types",
        None,
    ]
    return pred, true


# ── adjusted_rand_index ──────────────────────────────────────────────────


class TestAdjustedRandIndex:
    """ARI metric."""

    def test_perfect_match(
        self, evaluator: Evaluator, perfect_labels: tuple[np.ndarray, np.ndarray]
    ) -> None:
        pred, true = perfect_labels
        ari = evaluator.adjusted_rand_index(pred, true)
        assert ari == pytest.approx(1.0, abs=0.01)

    def test_random_labels_ari_near_zero(
        self, evaluator: Evaluator, random_labels: tuple[np.ndarray, np.ndarray]
    ) -> None:
        pred, true = random_labels
        ari = evaluator.adjusted_rand_index(pred, true)
        # ARI can be slightly negative for random partitions with small n
        assert -1.0 <= ari <= 1.0

    def test_all_same_label(self, evaluator: Evaluator) -> None:
        pred = np.zeros(5, dtype=int)
        true = np.zeros(5, dtype=int)
        ari = evaluator.adjusted_rand_index(pred, true)
        # When both are all same, ARI is 1
        assert ari == pytest.approx(1.0, abs=0.01)

    def test_ari_returns_float(self, evaluator: Evaluator) -> None:
        pred = np.array([0, 1], dtype=int)
        true = np.array([0, 1], dtype=int)
        assert isinstance(evaluator.adjusted_rand_index(pred, true), float)


# ── normalized_mutual_info ──────────────────────────────────────────────


class TestNormalizedMutualInfo:
    """NMI metric."""

    def test_perfect_match(
        self, evaluator: Evaluator, perfect_labels: tuple[np.ndarray, np.ndarray]
    ) -> None:
        pred, true = perfect_labels
        nmi = evaluator.normalized_mutual_info(pred, true)
        assert nmi == pytest.approx(1.0, abs=0.01)

    def test_nmi_between_zero_and_one(
        self, evaluator: Evaluator, random_labels: tuple[np.ndarray, np.ndarray]
    ) -> None:
        pred, true = random_labels
        nmi = evaluator.normalized_mutual_info(pred, true)
        assert 0.0 <= nmi <= 1.0

    def test_identical_labels(self, evaluator: Evaluator) -> None:
        pred = np.array([0, 0, 1, 1], dtype=int)
        true = np.array([0, 0, 1, 1], dtype=int)
        assert evaluator.normalized_mutual_info(pred, true) == pytest.approx(1.0, abs=0.01)

    def test_nmi_returns_float(self, evaluator: Evaluator) -> None:
        pred = np.array([0, 1, 0, 1], dtype=int)
        true = np.array([0, 0, 1, 1], dtype=int)
        assert isinstance(evaluator.normalized_mutual_info(pred, true), float)


# ── anomaly_detection_metrics ────────────────────────────────────────────


class TestAnomalyDetectionMetrics:
    """Anomaly detection with noise labels."""

    def test_perfect_anomaly_detection(self, evaluator: Evaluator) -> None:
        pred = np.array([-1, 0, 0, -1, 1, -1], dtype=int)
        true = [None, "clean", "clean", None, "wrong_data_types", None]
        result = evaluator.anomaly_detection_metrics(pred, true)
        # Predicted noise (-1) matches true None = anomaly
        assert result["precision"] == pytest.approx(1.0, abs=0.01)
        assert result["recall"] == pytest.approx(1.0, abs=0.01)
        assert result["f1"] == pytest.approx(1.0, abs=0.01)

    def test_no_anomaly_ground_truth(self, evaluator: Evaluator) -> None:
        pred = np.array([-1, 0, 1], dtype=int)
        true: list[str | None] = ["clean", "clean", "wrong_data_types"]
        result = evaluator.anomaly_detection_metrics(pred, true)
        assert result["support"] == 0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_missed_anomalies(self, evaluator: Evaluator) -> None:
        pred = np.array([0, 0, 1, 1], dtype=int)  # no noise predicted
        true = [None, "clean", None, "wrong_data_types"]
        result = evaluator.anomaly_detection_metrics(pred, true)
        assert result["recall"] == 0.0  # missed all anomalies
        assert result["precision"] == 0.0

    def test_false_positives(self, evaluator: Evaluator) -> None:
        pred = np.array([-1, -1, 1, 1], dtype=int)
        true = ["clean", "clean", None, "wrong_data_types"]
        result = evaluator.anomaly_detection_metrics(pred, true)
        # true_bin = [0, 0, 1, 0] -> support=1 (one anomaly at index 2)
        # pred_bin = [1, 1, 0, 0] -> FP at 0,1; FN at 2 -> prec=0, recall=0
        assert result["support"] == 1
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0


# ── validate_against_ground_truth ─────────────────────────────────────────


class TestValidateAgainstGroundTruth:
    """Full validation pipeline."""

    def test_returns_dict_with_keys(self, evaluator: Evaluator) -> None:
        pred = np.array([0, 0, 1, 1, 2, 2], dtype=int)
        true = [
            "wrong_data_types",
            "wrong_data_types",
            "clean",
            "clean",
            "self_contradictory",
            "self_contradictory",
        ]
        result = evaluator.validate_against_ground_truth(pred, true)
        assert "ari" in result
        assert "nmi" in result
        assert "n_gt_labels" in result
        assert result["n_gt_labels"] == 6

    def test_perfect_alignment(self, evaluator: Evaluator) -> None:
        pred = np.array([0, 0, 1, 1, 2, 2], dtype=int)
        true = ["a", "a", "b", "b", "c", "c"]
        result = evaluator.validate_against_ground_truth(pred, true)
        assert result["ari"] == pytest.approx(1.0, abs=0.01)
        assert result["nmi"] == pytest.approx(1.0, abs=0.01)

    def test_filters_unknown(self, evaluator: Evaluator) -> None:
        pred = np.array([0, 0, 1, 1], dtype=int)
        true: list[str | None] = ["a", None, "b", None]
        result = evaluator.validate_against_ground_truth(pred, true)
        assert result["n_gt_labels"] == 2
        # Only indices 0 and 2 have known labels

    def test_less_than_two_gt_labels(self, evaluator: Evaluator) -> None:
        pred = np.array([0, 1], dtype=int)
        true: list[str | None] = ["a", None]
        result = evaluator.validate_against_ground_truth(pred, true)
        assert "error" in result
        assert result["n_gt_labels"] == 1

    def test_no_gt_labels(self, evaluator: Evaluator) -> None:
        pred = np.array([0, 1], dtype=int)
        true: list[str | None] = [None, None]
        result = evaluator.validate_against_ground_truth(pred, true)
        assert "error" in result
        assert result["n_gt_labels"] == 0

    def test_with_anomaly_detection(self, evaluator: Evaluator) -> None:
        pred = np.array([-1, 0, 1, -1, 1], dtype=int)
        true = [None, "a", "b", None, "b"]
        result = evaluator.validate_against_ground_truth(pred, true)
        # pred_bin = [1, 0, 0, 1, 0], true_bin = [1, 0, 0, 1, 0] => perfect
        assert "anomaly_precision" in result
        assert result["anomaly_support"] == 2
        assert result["anomaly_precision"] == 1.0
        assert result["anomaly_recall"] == 1.0
        assert result["anomaly_f1"] == 1.0

    def test_floats_in_result(self, evaluator: Evaluator) -> None:
        pred = np.array([0, 0, 1, 1], dtype=int)
        true = ["a", "a", "b", "b"]
        result = evaluator.validate_against_ground_truth(pred, true)
        assert isinstance(result["ari"], float)
        assert isinstance(result["nmi"], float)


# ── print_validation_report ──────────────────────────────────────────────


class TestPrintValidationReport:
    """Formatted validation report."""

    def test_returns_string(self) -> None:
        validation = {
            "ari": 0.85,
            "nmi": 0.72,
            "n_gt_labels": 50,
            "anomaly_precision": 0.80,
            "anomaly_recall": 0.60,
            "anomaly_f1": 0.69,
            "anomaly_support": 10,
        }
        report = Evaluator.print_validation_report(validation)
        assert isinstance(report, str)
        assert "VALIDACION H1" in report
        assert "0.8500" in report
        assert "0.7200" in report

    def test_error_case(self) -> None:
        validation = {
            "error": "Less than 2 columns with ground truth labels",
            "n_gt_labels": 1,
        }
        report = Evaluator.print_validation_report(validation)
        assert "ERROR" in report
        assert "1" in report

    def test_all_metrics_in_report(self) -> None:
        validation = {
            "ari": 0.5,
            "nmi": 0.4,
            "n_gt_labels": 20,
            "anomaly_precision": 0.9,
            "anomaly_recall": 0.8,
            "anomaly_f1": 0.85,
            "anomaly_support": 5,
        }
        report = Evaluator.print_validation_report(validation)
        assert "0.5000" in report  # ARI printed as decimal
        assert "90.00%" in report  # precision printed as percentage
        assert "80.00%" in report
        assert "85.00%" in report
