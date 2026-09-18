"""
health_report.py — Schema health score and per-column structural recommendations.

Every other module in `evaluation/` answers "how good is this clustering?" This one
answers a different question: given a database (any Oracle database, not just the
243-column research corpus), how well-designed is it, and what should change?

The detector is a RandomForestClassifier fit on the 243 hand-labelled columns in
`output/manual_labels.csv` — not hand-written rules. `rule_engine.classify()` and its
detectors live on the `rule-engine` branch and stay there; a fresh set of structural
if/else checks here would just reinvent them under a different name. The classifier is
the only thing entitled to say "this looks like giant_table" — the recommendation TEXT
per label is static and written here, because that is presentation, not detection.

Feature space, and why it is not phi:
  FeatureBuilder.build() z-scores and block-normalizes whatever matrix is handed to
  it — the statistics are corpus-relative, computed fresh each call. That is fine for
  scoring one fixed corpus against itself, but it means phi from the reference corpus
  and phi from an independently-scored target database are not on the same scale
  unless both are normalized jointly. A RandomForest does not need that: every split
  is a per-feature threshold, so raw, unweighted, un-normalized blocks — the same
  columns in the same order every time a pickle is built by build_embeddings.py — are
  directly portable between any two databases built with the same --model/--semantic.
  See `build_raw_features`.

giant_table and table identity:
  AGENTS.md documents that most of the document embedding's edge on `giant_table`
  comes from the model recognising the two specific known giant tables by name, not
  from a generalisable "this table has too many columns" concept — none of the stored
  blocks carry table breadth, only per-column statistics. `build_raw_features` adds
  one extra raw feature, log1p(number of columns in the column's own table), read
  from the pickle's `schema` object, so a genuinely wide table in a new database has
  a chance of being recognised too.

Usage:
    model = fit_reference_model()
    result = score_reference(model)          # self-check, out-of-fold
    # or, for a different database already run through build_embeddings.py:
    result = score_target(model, "output/intermediate_other_db.pkl")
    print(format_health_report(result))
"""

from __future__ import annotations

import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from db_bad_clust.exceptions import BadDBError
from db_bad_clust.generation.ground_truth import MANUAL_LABEL_VOCABULARY

CV_FOLDS = 5
N_ESTIMATORS = 100

# Ordinal severity of the structural fix implied by each label — the user's own axis:
# how much of the schema has to be touched to fix it, not how "bad" it sounds.
#   0 = no fix needed
#   1 = cosmetic — rename only
#   2 = column-level — redefine one column's type/constraint
#   3 = table/schema-level — redesign spans multiple columns or tables
# Retune only here; nothing else in this module hardcodes these numbers.
SEVERITY: dict[str, int] = {
    "clean": 0,
    "reserved_words": 1,
    "inconsistent_naming": 1,
    "wrong_data_types": 2,
    "self_contradictory": 2,
    "impossible_data": 2,
    "self_referencing": 2,
    "giant_table": 3,
    "eav": 3,
    "polymorphic": 3,
}
MAX_SEVERITY = max(SEVERITY.values())

assert SEVERITY.keys() == MANUAL_LABEL_VOCABULARY, (
    "SEVERITY must cover exactly the manual label vocabulary"
)

RECOMMENDATIONS: dict[str, str] = {
    "clean": (
        "Sin cambios estructurales recomendados; la columna es consistente con su "
        "nombre, tipo y restricciones."
    ),
    "reserved_words": (
        "Renombrar la columna para evitar palabras reservadas de Oracle/SQL y no "
        "tener que escapar el identificador en cada consulta."
    ),
    "inconsistent_naming": (
        "Unificar la convención de nombres (prefijos, abreviaturas, mayúsculas/"
        "minúsculas) con el resto del esquema."
    ),
    "wrong_data_types": (
        "Revisar el tipo de dato declarado frente al contenido real de la columna "
        "y migrarlo al tipo que corresponda."
    ),
    "self_contradictory": (
        "El nombre y la definición estructural de la columna se contradicen; "
        "reconciliar el nombre, el tipo y las restricciones."
    ),
    "impossible_data": (
        "La definición actual permite valores sin sentido para el dominio (rangos, "
        "longitudes o combinaciones imposibles); acotar el tipo o añadir validaciones."
    ),
    "self_referencing": (
        "La columna referencia a su propia tabla de forma que puede producir ciclos "
        "o ambigüedad; documentar o rediseñar la relación jerárquica."
    ),
    "giant_table": (
        "La tabla tiene un número de columnas inusualmente alto; evaluar si agrupa "
        "varias entidades y dividirla en tablas más pequeñas y cohesivas."
    ),
    "eav": (
        "El patrón sugiere un modelo genérico entidad-atributo-valor; considerar "
        "modelar las entidades y atributos como columnas o tablas explícitas."
    ),
    "polymorphic": (
        "La columna parece referenciar a más de un tipo de entidad; separar en "
        "columnas o tablas específicas por tipo, o usar una tabla de asociación "
        "explícita."
    ),
}

assert RECOMMENDATIONS.keys() == MANUAL_LABEL_VOCABULARY, (
    "RECOMMENDATIONS must cover exactly the manual label vocabulary"
)


# ── Raw, portable feature matrix ────────────────────────────────────────────


def _table_column_counts(schema: Any) -> dict[str, int]:
    """Per-table column count, keyed by table name."""
    return {t.name: len(t.columns) for t in schema.tables}


def build_raw_features(data: dict[str, Any], *, include_table_width: bool = True) -> np.ndarray:
    """Concatenate e_text|e_type|e_rest|e_stat(|table width) at float64.

    Deliberately not FeatureBuilder.build(): that fusion's z-score/block-normalize
    statistics are computed from whatever matrix is passed to it, so its output is
    corpus-relative. Plain concatenation is portable between an independently-built
    reference pickle and target pickle, as long as both were built with the same
    --model/--semantic (build_embeddings.py).

    include_table_width appends log1p(number of columns in this column's table),
    read from data["schema"] — see the module docstring on why giant_table needs it.
    """
    blocks = [
        np.asarray(data[key], dtype=np.float64)
        for key in ("e_text", "e_type", "e_rest", "e_stat")
    ]
    if include_table_width:
        counts = _table_column_counts(data["schema"])
        widths = np.array(
            [[np.log1p(counts[key.split(".", 1)[0]])] for key in data["column_index"]]
        )
        blocks.append(widths)
    return np.hstack(blocks)


# ── Fit once, keep the model ────────────────────────────────────────────────


@dataclass
class HealthModel:
    """A RandomForest fit on the reference corpus, kept for `.predict()` on new data.

    Unlike `ml_baselines.random_forest_baseline`, which fits once, reads
    `feature_importances_`, and discards the classifier, this model is meant to be
    reused: `score_target` calls `.predict()`/`.predict_proba()` on it directly.
    """

    clf: Any
    label_support: dict[str, int]
    cv_f1_macro_mean: float
    cv_f1_macro_std: float
    include_table_width: bool
    reference_column_index: list[str]
    oof_predicted: list[str]
    oof_confidence: list[float]
    n_features: int


def fit_reference_model(
    reference_pickle: str | Path = "output/intermediate_docs.pkl",
    labels_path: str | Path = "output/manual_labels.csv",
    folds: int = CV_FOLDS,
    include_table_width: bool = True,
) -> HealthModel:
    """Train the classifier the health score and target-DB predictions come from.

    The labeled 243-column corpus is the only place labels ever come from; a target
    database contributes no labels, only unlabeled columns to run .predict() on.
    """
    import warnings

    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, cross_val_score

    from db_bad_clust.evaluation.cluster_scoring import load_manual_labels

    with open(reference_pickle, "rb") as fh:
        data = pickle.load(fh)
    column_index: list[str] = data["column_index"]
    truth = load_manual_labels(labels_path, column_index)

    raw = build_raw_features(data, include_table_width=include_table_width)

    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, class_weight="balanced", random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_scores = cross_val_score(clone(clf), raw, truth, cv=folds, scoring="f1_macro")
        oof_predicted = list(cross_val_predict(clone(clf), raw, truth, cv=folds))
        try:
            oof_proba = cross_val_predict(
                clone(clf), raw, truth, cv=folds, method="predict_proba"
            )
            oof_confidence = [float(row.max()) for row in oof_proba]
        except ValueError:
            # Defensive: a fold split too degenerate for predict_proba to align
            # classes. Rather than fail the whole report, mark confidence unknown —
            # the pessimistic NaN-handling in `_health_score` takes it from there.
            oof_confidence = [float("nan")] * len(truth)

    clf.fit(raw, truth)  # the persisted, full-data fit — used for score_target only

    return HealthModel(
        clf=clf,
        label_support=dict(Counter(truth)),
        cv_f1_macro_mean=float(cv_scores.mean()),
        cv_f1_macro_std=float(cv_scores.std()),
        include_table_width=include_table_width,
        reference_column_index=column_index,
        oof_predicted=oof_predicted,
        oof_confidence=oof_confidence,
        n_features=raw.shape[1],
    )


# ── Scoring ──────────────────────────────────────────────────────────────────


@dataclass
class HealthResult:
    """One column-by-column health verdict plus the aggregate score."""

    column_index: list[str]
    predicted: list[str]
    confidence: list[float]  # nan where unavailable
    severity: list[int]
    penalty: list[float]
    health_score: float
    mode: str  # "self_check (out-of-fold)" | "target (fitted model)"
    label_support: dict[str, int]
    low_support_labels: set[str] = field(default_factory=set)


def _health_score(severity: np.ndarray, confidence: np.ndarray) -> float:
    """penalty = severity * confidence; score = 100 * (1 - mean(penalty)/max)."""
    confidence = np.where(np.isnan(confidence), 1.0, confidence)  # NaN = pessimistic
    penalty = severity * confidence
    if len(penalty) == 0:
        return 100.0
    return float(np.clip(100.0 * (1.0 - penalty.mean() / MAX_SEVERITY), 0.0, 100.0))


def _low_support_labels(label_support: dict[str, int], folds: int) -> set[str]:
    return {label for label, n in label_support.items() if n < folds}


def _build_result(
    column_index: list[str],
    predicted: list[str],
    confidence: list[float],
    label_support: dict[str, int],
    mode: str,
    folds: int,
) -> HealthResult:
    severity = np.array([SEVERITY[label] for label in predicted], dtype=np.float64)
    conf_arr = np.array(confidence, dtype=np.float64)
    penalty = severity * np.where(np.isnan(conf_arr), 1.0, conf_arr)
    return HealthResult(
        column_index=column_index,
        predicted=predicted,
        confidence=confidence,
        severity=[int(s) for s in severity],
        penalty=[float(p) for p in penalty],
        health_score=_health_score(severity, conf_arr),
        mode=mode,
        label_support=label_support,
        low_support_labels=_low_support_labels(label_support, folds),
    )


def score_reference(model: HealthModel, folds: int = CV_FOLDS) -> HealthResult:
    """Self-check via the out-of-fold predictions computed while training.

    Scoring the corpus the model was trained on with `.predict()` reads close to a
    trivial in-sample fit and says nothing about generalisation — out-of-fold
    predictions are the honest number for the reference corpus itself.
    """
    return _build_result(
        column_index=model.reference_column_index,
        predicted=model.oof_predicted,
        confidence=model.oof_confidence,
        label_support=model.label_support,
        mode="self_check (out-of-fold)",
        folds=folds,
    )


def score_target(
    model: HealthModel,
    target_pickle: str | Path,
    folds: int = CV_FOLDS,
) -> HealthResult:
    """Score a new, disjoint database via the fully-fit model's predict/predict_proba.

    Raises BadDBError if the target's raw feature width does not match the model's
    training width — the two pickles must share --model/--semantic.
    """
    with open(target_pickle, "rb") as fh:
        data = pickle.load(fh)
    raw = build_raw_features(data, include_table_width=model.include_table_width)
    if raw.shape[1] != model.n_features:
        raise BadDBError(
            f"target feature width {raw.shape[1]} != reference training width "
            f"{model.n_features} — built with a different --model/--semantic?"
        )

    predicted = list(model.clf.predict(raw))
    proba = model.clf.predict_proba(raw)
    confidence = [float(row.max()) for row in proba]

    return _build_result(
        column_index=data["column_index"],
        predicted=predicted,
        confidence=confidence,
        label_support=model.label_support,
        mode="target (fitted model)",
        folds=folds,
    )


# ── Reporting ────────────────────────────────────────────────────────────────


def format_health_report(result: HealthResult, folds: int = CV_FOLDS) -> str:
    """Health score, per-label breakdown, and a mandatory low-support-labels section.

    A label with fewer training examples than `folds` (self_referencing has n=1 in
    the 243-column corpus) cannot have been learned reliably — its predictions must
    never be shown at the same confidence as a well-supported label. This mirrors
    cluster_scoring.py's own "say so wherever the number is quoted" convention for
    accuracy/F1.
    """
    lines: list[str] = []
    add = lines.append

    mode_note = (
        "out-of-fold predictions on the corpus the model was trained on — the "
        "honest self-check number, not an in-sample fit"
        if result.mode.startswith("self_check")
        else "predictions from the model fit on the full reference corpus — this "
        "database contributed no ground truth"
    )
    add(f"Mode: {result.mode}")
    add(f"  ({mode_note})")
    add("")
    add(f"Health score: {result.health_score:.1f} / 100")
    add(f"  {len(result.column_index)} columns scored")
    add("")

    by_label: dict[str, list[float]] = {}
    penalty_by_label: dict[str, float] = {}
    for label, conf, pen in zip(result.predicted, result.confidence, result.penalty, strict=True):
        by_label.setdefault(label, []).append(conf)
        penalty_by_label[label] = penalty_by_label.get(label, 0.0) + pen

    add(f"{'label':<24} {'count':>6} {'mean severity':>14} {'mean confidence':>16}")
    add("-" * 64)
    for label in sorted(by_label, key=lambda k: penalty_by_label[k], reverse=True):
        confs = np.array(by_label[label], dtype=np.float64)
        mean_conf = float(np.nanmean(confs)) if len(confs) else float("nan")
        add(
            f"{label:<24} {len(by_label[label]):>6} {SEVERITY[label]:>14} "
            f"{mean_conf:>16.3f}"
        )
    add("")

    if result.low_support_labels:
        add("LOW-SUPPORT LABELS — trained on fewer examples than the CV fold count")
        add("-" * 64)
        for label in sorted(result.low_support_labels):
            n = result.label_support.get(label, 0)
            flagged_columns = [
                col
                for col, pred in zip(result.column_index, result.predicted, strict=True)
                if pred == label
            ]
            add(f"  {label}: trained on n={n} (< {folds} folds) — cannot be learned reliably")
            if flagged_columns:
                add(f"    predicted here: {', '.join(flagged_columns[:10])}")
                if len(flagged_columns) > 10:
                    add(f"    ... and {len(flagged_columns) - 10} more")
        add("")

    add("Recommendations are generic structural advice, not generated DDL —")
    add("this branch has no DDLGenerator; see the rule-engine branch for that.")
    return "\n".join(lines)


def write_health_csv(path: str | Path, result: HealthResult) -> None:
    """Per-column health verdict, matching cluster_scoring.write_per_column_csv's style."""
    import csv

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["column", "predicted_label", "confidence", "severity", "low_support", "recommendation"]
        )
        for col, label, conf, sev in zip(
            result.column_index, result.predicted, result.confidence, result.severity, strict=True
        ):
            writer.writerow(
                [
                    col,
                    label,
                    "" if np.isnan(conf) else round(float(conf), 4),
                    sev,
                    "yes" if label in result.low_support_labels else "no",
                    RECOMMENDATIONS[label],
                ]
            )
