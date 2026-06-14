"""
test_reporter.py — Tests for reporter.py module
"""

import pytest
from reporter import EvaluationReporter


class TestInternalMetricsReport:
    """Tests for internal metrics report."""

    def test_returns_string(self):
        """Report should be a string."""
        metrics = {
            "silhouette": 0.5,
            "davies_bouldin": 0.8,
            "calinski_harabasz": 150.0,
            "n_clusters": 3,
            "n_samples": 100,
            "n_noise": 5,
        }
        report = EvaluationReporter.internal_metrics_report(metrics)
        assert isinstance(report, str)

    def test_contains_metrics(self):
        """Report should contain metric values."""
        metrics = {
            "silhouette": 0.5,
            "davies_bouldin": 0.8,
            "calinski_harabasz": 150.0,
            "n_clusters": 3,
            "n_samples": 100,
            "n_noise": 5,
        }
        report = EvaluationReporter.internal_metrics_report(metrics)
        assert "Silhouette Score" in report
        assert "Davies-Bouldin Index" in report
        assert "Calinski-Harabasz Index" in report

    def test_error_handling(self):
        """Report handles error in metrics."""
        metrics = {"error": "No samples"}
        report = EvaluationReporter.internal_metrics_report(metrics)
        assert "ERROR" in report

    def test_with_cross_table(self):
        """Report includes cross-table analysis."""
        metrics = {
            "silhouette": 0.5,
            "davies_bouldin": 0.8,
            "calinski_harabasz": 150.0,
            "n_clusters": 3,
            "n_samples": 100,
            "n_noise": 0,
        }
        cross_table = {
            "average_purity": 0.85,
            "table_purity": {
                "EMPLOYEES": {"total_columns": 5, "purity": 0.8, "majority_cluster": 0},
            },
        }
        report = EvaluationReporter.internal_metrics_report(metrics, cross_table=cross_table)
        assert "Distribucion por Tabla" in report
        assert "EMPLOYEES" in report


class TestValidationReport:
    """Tests for validation report."""

    def test_returns_string(self):
        """Report should be a string."""
        validation = {
            "ari": 0.5,
            "nmi": 0.6,
            "n_gt_labels": 50,
            "anomaly_precision": 0.7,
            "anomaly_recall": 0.8,
            "anomaly_f1": 0.75,
            "anomaly_support": 10,
        }
        report = EvaluationReporter.validation_report(validation)
        assert isinstance(report, str)

    def test_contains_validation_metrics(self):
        """Report contains validation metrics."""
        validation = {
            "ari": 0.5,
            "nmi": 0.6,
            "n_gt_labels": 50,
            "anomaly_precision": 0.7,
            "anomaly_recall": 0.8,
            "anomaly_f1": 0.75,
            "anomaly_support": 10,
        }
        report = EvaluationReporter.validation_report(validation)
        assert "Adjusted Rand Index" in report
        assert "Normalized Mutual Info" in report

    def test_with_structural_anomalies(self):
        """Report includes structural anomalies when present."""
        validation = {
            "ari": 0.5,
            "nmi": 0.6,
            "n_gt_labels": 50,
            "anomaly_precision": 0.7,
            "anomaly_recall": 0.8,
            "anomaly_f1": 0.75,
            "anomaly_support": 10,
            "structural_anomalies": 5,
            "structural_anomaly_ratio": 0.05,
            "centroid_mean_distance": 2.5,
            "centroid_std_distance": 0.8,
        }
        report = EvaluationReporter.validation_report(validation)
        assert "Structural anomalies" in report
        assert "5" in report

    def test_error_handling(self):
        """Report handles error in validation."""
        validation = {"error": "Less than 2 columns", "n_gt_labels": 0}
        report = EvaluationReporter.validation_report(validation)
        assert "ERROR" in report
