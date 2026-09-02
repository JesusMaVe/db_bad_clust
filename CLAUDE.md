# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` in this same repo has the full detail this file only summarizes: the measured
results, the invariants learned from getting them wrong, and the Oracle-specific gotchas.
Read it before touching `features/feature_builder.py` or `features/text_preprocessor.py` —
several behaviors there look like details and are the finding.

## What this is

**This branch is the research: unsupervised detection of Oracle schema anti-patterns using
sentence embeddings.** BERT is the method, not a comparison arm. Columns are described as
Spanish sentences, embedded, fused with structural features, reduced and clustered; the
clustering is scored against 243 hand-labelled columns in `output/manual_labels.csv`.

**The rule engine lives on the `rule-engine` branch**, together with the Oracle audit product
(`cli audit`, `Recommender`, `DDLGenerator`) and its 472 tests. It is not part of `main` and
should not be reintroduced here — if you find yourself wanting `rule_engine.classify()`, you
are on the wrong branch.

## Commands

```bash
docker compose up -d                      # Oracle 23c, healthcheck ~30s
.venv/bin/python -m pip install -e ".[dev]"
```

Always use `.venv/bin/python -m <cmd>` (not bare `pytest`/`jupyter`/`ruff`) — venv
entrypoint shebangs are stale.

```bash
# Score configurations against the manual labels. No database needed.
.venv/bin/python -m db_bad_clust.cli experiment --sweep
.venv/bin/python -m db_bad_clust.cli experiment --without-giants \
    --ablation output/intermediate_02.pkl output/intermediate_docs.pkl

# Rebuild the feature blocks from live Oracle metadata. Needs the container up.
.venv/bin/python scripts/apply_comments.py            # once: puts comments in the DB
.venv/bin/python scripts/build_embeddings.py --dry-run
.venv/bin/python scripts/build_embeddings.py --output output/intermediate_docs.pkl
```

```bash
.venv/bin/python -m pytest tests/ -v      # 372 tests, no DB required
.venv/bin/python -m ruff check src tests scripts   # lint-clean
```

`pyproject.toml` sets `pythonpath = ["src"]` for pytest. Notebooks and `scripts/` do their own
`sys.path.insert` — the editable-install `.pth` mechanism has been unreliable in at least one
dev environment here; don't remove the explicit inserts.

## Architecture

```
src/db_bad_clust/
├── cli.py         `experiment` — the surface that produces every reported number
├── data/          Oracle in, dataclasses out: db_connector, schema_extractor
├── features/      text_preprocessor (+ build_document), structural_encoder,
│                  bert_embedder, semantic_anchors, feature_builder
├── clustering/    dimensionality_reducer, cluster_engine
├── evaluation/    cluster_scoring (both rulers), experiments (sweeps, ablations,
│                  diagnostics), validation (ARI/NMI), ml_baselines
└── generation/    anti_patterns (table/column comments), ground_truth (loads +
                   validates the manual labels), label_export
scripts/           apply_comments.py, build_embeddings.py
notebooks/         01, 02 — the original pipeline; keep them, they produce
                   output/intermediate_02.pkl, the historical baseline artifact
tests/             pytest, entirely mock-based — no DB, no model download
```

## The three results this branch rests on

**1. `alpha` never weighted anything.** `FeatureBuilder._zscore` normalizes per dimension, so a
block's *total* variance equals its dimension count — 384 for the embeddings against 7 effective
for types and 1 for constraints. Multiplying by `alpha` afterwards cannot undo that. At a
nominal `alpha=0.15` the embeddings held **89%** of the variance; at 0.30, **98%**. The
documented "BERT is harmful, ARI 0.5008 → 0.30" was a comparison between "no embeddings" and
"embeddings and almost nothing else". **Use `normalize="block"` (the default). `"zscore"` exists
only to reproduce the old numbers.**

**2. The structural baseline cannot individuate the corpus.** With `alpha=0`, the 243 columns
collapse into **24 distinct vectors** — 97% of rows duplicate another, and the largest tie group
holds 108 columns. Columns sharing a vector cannot be given different labels by any algorithm.
Its ARI is also not reproducible: 0.1877 in float32 against 0.5031 in float64. `load_dataset`
pins float64 deliberately; `representation_degeneracy` and `precision_sensitivity` measure both.

**3. What the encoder is given to read is the whole game.** `process()` yields
"empleados: fecha nacimiento" — no type in it, so no encoder can tell `FECHA_INGRESO DATE` from
`FECHA_INGRESO VARCHAR2(20)`. `build_document()` writes the type and constraints out in words.
On the 153 columns that remain once the two giant tables are excluded, at `alpha=1.00`:

| representation      |     ARI |    NMI |    AMI | Accuracy | F1-macro |
| ------------------- | ------: | -----: | -----: | -------: | -------: |
| structure only      |  0.0657 | 0.2432 | 0.1212 |   0.6078 |   0.2002 |
| name only (old)     | -0.0538 | 0.2043 | 0.0747 |   0.5425 |   0.1346 |
| **column document** |  0.0802 | 0.3788 | 0.2553 |   0.6797 |   0.3781 |

The document wins all five. The old name-only embedding scores **negative** ARI — worse than
chance. The difference was never BERT; it was the input.

## Invariants

- **`FeatureBuilder` defaults to `normalize="block"`.** Reintroducing per-dimension-only
  scaling silently makes every weight meaningless again.
- **`build_document` writes Spanish; use `TextPreprocessor.for_documents()`** so names are
  expanded with `ABBREVIATIONS_ES`. The default dictionary expands into English, which is right
  for `process()` and wrong inside a Spanish sentence — and it hits the two most frequent tokens
  in the schema, `col` (53 columns) and `id` (24).
- **The anchors read the column *name*, never the document.** The document states the type,
  which is the answer the mismatch feature is being asked for.
- **Report AMI, not NMI, across rows with different `k`.** Configurations here produce between
  2 and 19 clusters and NMI rises with `k` on its own.
- **`giant_table` is a property of the table, not the column** — 90 of 243 columns, all from
  `TABLA_BASE_DATOS` (40) and `BACKUP_DATOS` (50). Quote `--without-giants` alongside the full
  corpus; the full-corpus gain is partly table-identity leakage through the table comment, and
  `build_embeddings.py --no-table-comment` measures how much.
- **Accuracy/F1 for a clustering are an upper bound** — majority-vote naming uses the ground
  truth the pipeline never saw. Say so wherever they are quoted.
- **Never overwrite `output/intermediate_02.pkl`.** It is the historical baseline; an
  experiment that overwrites its own baseline cannot be checked.

- **The `--mismatch` scalar is a mixed result, not a clean win** — it lifts F1-macro
  (majority-vote naming can use it to name a cluster better) but slightly lowers ARI/AMI (it
  does not change which columns end up together). Keep it opt-in.

## Gotchas

- Table names are case-sensitive in Oracle — always double-quote.
- Oracle allows only one LONG column per table (ORA-01754); the catalog uses CLOB/BLOB.
- `apply_comments.py` applies 77 comments (23 tables, 54 columns). Before it ran, all 243
  columns had `comments = None` and the encoder had no natural-language input at all.
