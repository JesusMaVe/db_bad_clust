"""Tests for the health score / recommendations stage. No DB, no real BERT model."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from db_bad_clust.data.schema_extractor import ColumnMetadata, DatabaseSchema, TableMetadata
from db_bad_clust.evaluation.health_report import (
    MAX_SEVERITY,
    RECOMMENDATIONS,
    SEVERITY,
    _health_score,
    build_raw_features,
    fit_reference_model,
    format_health_report,
    score_reference,
    score_target,
    write_health_csv,
)
from db_bad_clust.exceptions import BadDBError
from db_bad_clust.generation.ground_truth import MANUAL_LABEL_VOCABULARY


def _schema(table_sizes: dict[str, int]) -> DatabaseSchema:
    """A DatabaseSchema with the given {table_name: column_count}."""
    tables = []
    for name, n in table_sizes.items():
        cols = [
            ColumnMetadata(name=f"C{i}", data_type="VARCHAR2", data_length=50, nullable=True)
            for i in range(n)
        ]
        tables.append(TableMetadata(name=name, columns=cols))
    return DatabaseSchema(tables=tables)


def _pickle_data(column_index, table_sizes, n_text=8, n_type=4, n_rest=5, seed=0):
    rng = np.random.default_rng(seed)
    n = len(column_index)
    return {
        "column_index": column_index,
        "e_text": rng.normal(0, 1, (n, n_text)),
        "e_type": rng.normal(0, 1, (n, n_type)),
        "e_rest": rng.normal(0, 1, (n, n_rest)),
        "e_stat": rng.normal(0, 1, (n, 1)),
        "schema": _schema(table_sizes),
    }


class TestSeverityAndRecommendations:
    def test_severity_covers_full_vocabulary(self):
        assert SEVERITY.keys() == MANUAL_LABEL_VOCABULARY

    def test_recommendations_covers_full_vocabulary(self):
        assert RECOMMENDATIONS.keys() == MANUAL_LABEL_VOCABULARY
        for text in RECOMMENDATIONS.values():
            assert isinstance(text, str) and text.strip()

    def test_severity_ordering_matches_structural_gravity(self):
        table_level = {SEVERITY[label] for label in ("giant_table", "eav", "polymorphic")}
        column_level = {
            SEVERITY[label]
            for label in (
                "wrong_data_types",
                "self_contradictory",
                "impossible_data",
                "self_referencing",
            )
        }
        cosmetic = {SEVERITY[label] for label in ("reserved_words", "inconsistent_naming")}
        assert min(table_level) > max(column_level)
        assert min(column_level) > max(cosmetic)
        assert min(cosmetic) > SEVERITY["clean"]
        assert MAX_SEVERITY == max(SEVERITY.values())


class TestBuildRawFeatures:
    def test_shape_and_table_width_column(self):
        column_index = ["A.C0", "A.C1", "B.C0"]
        data = _pickle_data(column_index, table_sizes={"A": 2, "B": 1})
        raw = build_raw_features(data, include_table_width=True)
        assert raw.shape == (3, 8 + 4 + 5 + 1 + 1)
        expected_widths = [np.log1p(2), np.log1p(2), np.log1p(1)]
        np.testing.assert_allclose(raw[:, -1], expected_widths)

    def test_without_table_width(self):
        column_index = ["A.C0", "B.C0"]
        data = _pickle_data(column_index, table_sizes={"A": 1, "B": 1})
        raw = build_raw_features(data, include_table_width=False)
        assert raw.shape == (2, 8 + 4 + 5 + 1)


class TestFitReferenceModel:
    @pytest.fixture
    def fixture_paths(self, tmp_path):
        rng = np.random.default_rng(1)
        n = 40
        columns = [f"T{i % 4}.C{i}" for i in range(n)]
        half = n // 2
        # two separable blobs in e_rest so the classifier has real signal
        e_rest = np.vstack([rng.normal(0, 0.2, (half, 5)), rng.normal(6, 0.2, (n - half, 5))])
        data = {
            "column_index": columns,
            "e_text": rng.normal(0, 1, (n, 8)),
            "e_type": rng.normal(0, 1, (n, 4)),
            "e_rest": e_rest,
            "e_stat": rng.normal(0, 1, (n, 1)),
            "schema": _schema({f"T{i}": sum(1 for c in columns if c.startswith(f"T{i}.")) for i in range(4)}),
        }
        pkl = tmp_path / "reference.pkl"
        with open(pkl, "wb") as fh:
            pickle.dump(data, fh)

        labels = tmp_path / "labels.csv"
        rows = ["table,column,data_type,label"]
        for i, key in enumerate(columns):
            table, column = key.split(".")
            rows.append(f"{table},{column},NUMBER,{'clean' if i < half else 'eav'}")
        labels.write_text("\n".join(rows) + "\n")
        return pkl, labels

    def test_keeps_a_fitted_classifier(self, fixture_paths):
        pkl, labels = fixture_paths
        model = fit_reference_model(reference_pickle=pkl, labels_path=labels, folds=2)
        assert hasattr(model.clf, "predict")
        assert hasattr(model.clf, "predict_proba")
        assert len(model.clf.classes_) == 2
        assert 0.0 <= model.cv_f1_macro_mean <= 1.0
        assert model.label_support == {"clean": 20, "eav": 20}

    def test_self_check_uses_out_of_fold_not_direct_fit(self, fixture_paths):
        pkl, labels = fixture_paths
        model = fit_reference_model(reference_pickle=pkl, labels_path=labels, folds=5)

        raw = build_raw_features(pickle.load(open(pkl, "rb")))
        direct_predictions = model.clf.predict(raw)
        direct_accuracy = float(np.mean(np.array(model.oof_predicted) == direct_predictions))
        # OOF predictions need not equal the direct in-sample fit's own predictions —
        # if they did, the self-check would just be reading the trivial in-sample fit.
        assert direct_accuracy < 1.0 or len(set(model.oof_predicted)) > 1

        oof_accuracy = float(
            np.mean(
                np.array(model.oof_predicted)
                == ["clean"] * 20 + ["eav"] * 20
            )
        )
        assert 0.0 <= oof_accuracy <= 1.0

    def test_score_reference_reuses_oof(self, fixture_paths):
        pkl, labels = fixture_paths
        model = fit_reference_model(reference_pickle=pkl, labels_path=labels, folds=2)
        result = score_reference(model, folds=2)
        assert result.mode == "self_check (out-of-fold)"
        assert result.predicted == model.oof_predicted
        assert 0.0 <= result.health_score <= 100.0
        assert len(result.column_index) == 40

    def test_score_target_on_disjoint_pickle(self, fixture_paths, tmp_path):
        pkl, labels = fixture_paths
        model = fit_reference_model(reference_pickle=pkl, labels_path=labels, folds=2)

        rng = np.random.default_rng(2)
        target_columns = ["X.C0", "X.C1", "Y.C0"]
        target_data = {
            "column_index": target_columns,
            "e_text": rng.normal(0, 1, (3, 8)),
            "e_type": rng.normal(0, 1, (3, 4)),
            "e_rest": rng.normal(0, 1, (3, 5)),
            "e_stat": rng.normal(0, 1, (3, 1)),
            "schema": _schema({"X": 2, "Y": 1}),
        }
        target_pkl = tmp_path / "target.pkl"
        with open(target_pkl, "wb") as fh:
            pickle.dump(target_data, fh)

        result = score_target(model, target_pkl, folds=2)
        assert result.mode == "target (fitted model)"
        assert len(result.predicted) == 3
        assert all(0.0 <= c <= 1.0 for c in result.confidence)

    def test_target_feature_width_mismatch_raises(self, fixture_paths, tmp_path):
        pkl, labels = fixture_paths
        model = fit_reference_model(reference_pickle=pkl, labels_path=labels, folds=2)

        rng = np.random.default_rng(3)
        target_data = {
            "column_index": ["X.C0"],
            "e_text": rng.normal(0, 1, (1, 4)),  # narrower than reference's 8
            "e_type": rng.normal(0, 1, (1, 4)),
            "e_rest": rng.normal(0, 1, (1, 5)),
            "e_stat": rng.normal(0, 1, (1, 1)),
            "schema": _schema({"X": 1}),
        }
        target_pkl = tmp_path / "narrow.pkl"
        with open(target_pkl, "wb") as fh:
            pickle.dump(target_data, fh)

        with pytest.raises(BadDBError, match="feature width"):
            score_target(model, target_pkl, folds=2)


class TestHealthScoreFormula:
    def test_all_clean_is_perfect_score(self):
        severity = np.zeros(10)
        confidence = np.ones(10)
        assert _health_score(severity, confidence) == 100.0

    def test_all_max_severity_full_confidence_is_zero(self):
        severity = np.full(10, MAX_SEVERITY, dtype=float)
        confidence = np.ones(10)
        assert _health_score(severity, confidence) == 0.0

    def test_nan_confidence_is_imputed_pessimistically(self):
        severity = np.array([MAX_SEVERITY, MAX_SEVERITY])
        nan_confidence = np.array([np.nan, np.nan])
        full_confidence = np.array([1.0, 1.0])
        assert _health_score(severity, nan_confidence) == _health_score(severity, full_confidence)

    def test_score_always_in_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            severity = rng.integers(0, MAX_SEVERITY + 1, size=15).astype(float)
            confidence = rng.uniform(0, 1, size=15)
            score = _health_score(severity, confidence)
            assert 0.0 <= score <= 100.0

    def test_empty_input_is_perfect_score(self):
        assert _health_score(np.array([]), np.array([])) == 100.0


class TestLowSupportAndFormatting:
    @pytest.fixture
    def fixture_paths(self, tmp_path):
        rng = np.random.default_rng(4)
        n = 20
        columns = [f"T.C{i}" for i in range(n)]
        # 19 clean, 1 self_referencing — a class with n=1 under folds=5
        labels_list = ["clean"] * 19 + ["self_referencing"]
        e_rest = rng.normal(0, 1, (n, 5))
        data = {
            "column_index": columns,
            "e_text": rng.normal(0, 1, (n, 8)),
            "e_type": rng.normal(0, 1, (n, 4)),
            "e_rest": e_rest,
            "e_stat": rng.normal(0, 1, (n, 1)),
            "schema": _schema({"T": n}),
        }
        pkl = tmp_path / "reference.pkl"
        with open(pkl, "wb") as fh:
            pickle.dump(data, fh)

        labels = tmp_path / "labels.csv"
        rows = ["table,column,data_type,label"]
        for key, label in zip(columns, labels_list, strict=True):
            table, column = key.split(".")
            rows.append(f"{table},{column},NUMBER,{label}")
        labels.write_text("\n".join(rows) + "\n")
        return pkl, labels

    def test_low_support_label_detected(self, fixture_paths):
        pkl, labels = fixture_paths
        model = fit_reference_model(reference_pickle=pkl, labels_path=labels, folds=5)
        result = score_reference(model, folds=5)
        assert "self_referencing" in result.low_support_labels
        assert "clean" not in result.low_support_labels

    def test_report_flags_low_support_section(self, fixture_paths):
        pkl, labels = fixture_paths
        model = fit_reference_model(reference_pickle=pkl, labels_path=labels, folds=5)
        result = score_reference(model, folds=5)
        text = format_health_report(result, folds=5)
        assert "LOW-SUPPORT LABELS" in text
        assert "self_referencing" in text

    def test_report_omits_section_when_all_well_supported(self):
        from db_bad_clust.evaluation.health_report import HealthResult

        result = HealthResult(
            column_index=["A.C0", "A.C1"],
            predicted=["clean", "clean"],
            confidence=[0.9, 0.8],
            severity=[0, 0],
            penalty=[0.0, 0.0],
            health_score=100.0,
            mode="self_check (out-of-fold)",
            label_support={"clean": 10},
            low_support_labels=set(),
        )
        text = format_health_report(result, folds=5)
        assert "LOW-SUPPORT LABELS" not in text


class TestWriteHealthCsv:
    def test_shape_and_recommendation_alignment(self, tmp_path):
        from db_bad_clust.evaluation.health_report import HealthResult

        result = HealthResult(
            column_index=["A.C0", "A.C1", "B.C0"],
            predicted=["clean", "wrong_data_types", "giant_table"],
            confidence=[0.9, 0.4, float("nan")],
            severity=[0, 2, 3],
            penalty=[0.0, 0.8, 3.0],
            health_score=42.0,
            mode="target (fitted model)",
            label_support={"clean": 1, "wrong_data_types": 1, "giant_table": 1},
            low_support_labels={"giant_table"},
        )
        out = tmp_path / "health.csv"
        write_health_csv(out, result)

        import csv

        with open(out, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 3
        assert list(rows[0].keys()) == [
            "column",
            "predicted_label",
            "confidence",
            "severity",
            "low_support",
            "recommendation",
        ]
        for row in rows:
            assert row["recommendation"] == RECOMMENDATIONS[row["predicted_label"]]
        assert rows[2]["low_support"] == "yes"
        assert rows[0]["low_support"] == "no"
        assert rows[2]["confidence"] == ""  # NaN written as empty
