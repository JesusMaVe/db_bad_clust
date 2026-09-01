# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` in this same repo has the full detail this file only summarizes: critical
invariants learned from past bugs (issues #1-#7), current metrics, and Oracle-specific
gotchas. Read it before touching `rules/rule_engine.py` or the `generation/` catalog —
several behaviors there look like bugs but are deliberate.

## What this is

Detects schema anti-patterns in Oracle databases two ways that are meant to cross-check
each other: a rule engine (`src/db_bad_clust/rules/rule_engine.py`) that classifies
columns/tables against hand-written heuristics, and an unsupervised clustering pipeline
(BERT embeddings + structural features) that should independently rediscover the same
groupings without labels. `generation/anti_patterns.py` defines a synthetic "bad schema"
catalog that seeds the real Oracle instance and doubles as ground truth.

## Commands

```bash
docker compose up -d                         # Oracle 23c, healthcheck ~30s
.venv/bin/python -m pip install -e ".[dev]"  # installs db_bad_clust + deps from pyproject.toml
```

Dependencies are split: the base install is only `oracledb` + `pyyaml` (all the audit path
needs), and everything ML — torch, transformers, umap-learn, hdbscan, sklearn, numpy — lives in
the `research` extra. `[dev]` pulls `[research]`. Don't move an ML import into the audit path
without moving its dependency too; the point of the split is that auditing a schema does not
download ~2GB for the branch the tool doesn't use.

```bash
.venv/bin/python -m db_bad_clust.cli audit --sql output/fix.sql   # live Oracle audit
.venv/bin/python -m db_bad_clust.cli compare --sweep              # rule engine vs clustering, no DB
```

Always use `.venv/bin/python -m <cmd>` (not bare `pytest`/`jupyter`/`ruff`) — venv
entrypoint shebangs are stale.

```bash
.venv/bin/python -m pytest tests/ -v                                          # 472 tests, no DB required
.venv/bin/python -m pytest tests/test_rule_engine.py::TestSchemaSpySignals -q # single test class
.venv/bin/python -m ruff check src tests scripts                              # 11 pre-existing errors (RUF012 mutable class defaults); new code should be lint-clean
.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/01_data_preparation.ipynb  # headless notebook run; needs Oracle up
```

`pyproject.toml` sets `pythonpath = ["src"]` for pytest, so tests resolve `db_bad_clust`
without relying on the editable install. Notebooks and `scripts/` do their own
`sys.path.insert(0, ".../src")` for the same reason — the editable-install `.pth`
mechanism has been unreliable in at least one dev environment here; don't remove the
explicit inserts on the assumption `pip install -e .` alone is enough.

## Architecture

```
src/db_bad_clust/
├── cli.py         the product surface: `audit` (live Oracle) and `compare` (no DB)
├── data/          Oracle in, dataclasses out: db_connector, schema_extractor
├── features/      text_preprocessor, structural_encoder, bert_embedder, feature_builder
├── clustering/    dimensionality_reducer, cluster_engine
├── rules/         rule_engine (detection — the single home for keyword/reserved-word
│                  constants), recommender, recommendation_reporter
├── evaluation/    metrics (internal clustering quality), validation (external,
│                  ground-truth comparison), anomaly, head_to_head (rule engine vs
│                  clustering, both scored with both rulers)
└── generation/    anti_patterns (synthetic catalog), schema_adapter (catalog →
                   DatabaseSchema), ddl_generator, ground_truth (owns RULE_TO_MANUAL label
                   mapping + MANUAL_LABEL_VOCABULARY), label_export
scripts/           apply_comments.py — one-shot DB script, not imported by the package
notebooks/         01-04, the research record (see below)
tests/             pytest, entirely mock-based — no DB needed
```

**`cli.py` is the product; the notebooks are the research record.** `audit` wires
extractor → rule_engine → recommender → ddl_generator, which is the only caller
`ddl_generator` has — it sat uninvoked before. The notebooks still run the full ML
pipeline in order, each pickling its output to `output/` for the next:

1. `01_data_preparation.ipynb` — extraction + preprocessing + structural encoding
2. `02_ml_embeddings.ipynb` — sentence embeddings + feature building + dimensionality reduction
Notebooks 03/04 were deleted — `audit` and `compare` cover everything they did, and their
one unique piece (the Random Forest benchmark) moved to `evaluation/ml_baselines.py`.
01 needs Oracle up; 02 reads 01's pickle. Do NOT delete these two: they are the only
producers of `output/intermediate_02.pkl`, which `compare` reads, so removing them makes
the evidence unreproducible.

**Keyword and label constants have exactly one home each** — this used to be duplicated
by hand across files and drifted; it no longer is, and reintroducing a local copy
un-does that fix:

- SQL keyword sets (`SQL_RESERVED_WORDS`, `_kw_match`, `NUMBER_KEYWORDS`,
  `DATE_KEYWORDS_HIGH`) live in `rules/rule_engine.py`; `generation/ddl_generator.py`
  imports them rather than keeping a local copy.
- The rule-label → manual-ground-truth-label mapping (`RULE_TO_MANUAL`) and the manual
  label vocabulary (`MANUAL_LABEL_VOCABULARY`) live in `generation/ground_truth.py`;
  `rules/recommender.py` and `generation/label_export.py` import them.

**Ground truth and detection deliberately share code.** `ground_truth.build_ground_truth_map()`
runs the same `rule_engine.classify()` used for real detection, so "accuracy" against
this ground truth is partly circular (see AGENTS.md for the honest, manually-labeled
comparison). Don't treat the aligned-GT metrics as an unbiased accuracy figure.

There is no *evaluation* facade — `evaluator.py`/`reporter.py` were deleted (pure
pass-throughs over `evaluation/metrics.py`, `evaluation/validation.py`,
`evaluation/anomaly.py`); call those modules directly instead of looking for a wrapper.
`cli.py` is a real entry point, not a facade: it orchestrates a pipeline no single module
owns.

**The clustering branch lost, and the numbers are reproducible.** `evaluation/head_to_head.py`
scores both branches with both rulers against the 243 manual labels — a rule engine's labels
are also a partition (so it takes ARI/NMI), and clusters become labels by majority vote (so
they take accuracy/F1). Measured:

| Branch                  |    ARI |    NMI | Accuracy | F1-macro |
| ----------------------- | -----: | -----: | -------: | -------: |
| Rule engine (A)         | 0.7667 | 0.8058 |   0.8930 |   0.8467 |
| Clustering (B)          | 0.5008 | 0.4199 |   0.7078 |   0.2639 |

Two things to preserve when touching this file: **B's accuracy/F1 are an upper bound**
(majority-vote naming leaks ground truth to B, which A never gets), and **`_run_branch_a`
must apply `RULE_TO_MANUAL`** — the engine emits finer labels (`bad_boolean`,
`date_as_text`) than the manual vocabulary, and skipping the translation silently drops A
from 0.8930 to 0.8560. B scores 0.000 on six of ten classes because a cluster is named for
its majority, so a minority anti-pattern is never pronounced. `alpha_sweep()` shows BERT is
not merely useless but harmful: ARI 0.5008 at α=0 falls to ~0.30 the moment it is switched
on. Full write-up: `docs/veredicto_reglas_vs_ml.html`.

`generation/schema_generator.py` and `generation/benchmark.py` were deleted — synthetic-schema
fabrication that no notebook or script called, obsolete once the manual labels existed. The one
piece still needed, `to_database_schema()`, now lives in `generation/schema_adapter.py`
(verified: `build_ground_truth_map()` output is byte-identical before and after).
