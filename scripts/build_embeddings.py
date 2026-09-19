"""
build_embeddings.py — Rebuild the feature blocks from live Oracle metadata.

Notebook 02 embedded `TextPreprocessor.process` output: a bare noun phrase like
"empleados: fecha nacimiento", with no type in it. This script embeds
`TextPreprocessor.build_document` output instead — a Spanish sentence carrying
the declared type, the constraints and the data-dictionary comments — so the
encoder can represent the mismatch between what a column is called and how it
is stored.

It writes a NEW pickle and never touches `output/intermediate_02.pkl`: the old
artifact is what reproduces the historical numbers, and an experiment that
overwrites its own baseline cannot be checked.

Every pickle also carries `e_conflict`, the raw name-expectation-minus-declared-
type block (features/semantic_anchors.py, ConflictBlock), in hard mode with the
default anchors, and `e_conflict_variants`, the same block under each wording
in TYPE_FAMILY_ANCHOR_VARIANTS — the input to `cli experiment --robustness`.
Both cost one encoder pass over the bare column names and are weighted by
`zeta`, 0.0 by default, so storing them changes no existing experiment.

Usage:
    .venv/bin/python scripts/build_embeddings.py --output output/intermediate_docs.pkl
    .venv/bin/python scripts/build_embeddings.py --dry-run          # print 5 documents, no model
    .venv/bin/python scripts/build_embeddings.py --from-pickle output/intermediate_docs.pkl \
        --output output/intermediate_docs.pkl                     # re-embed without Oracle
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from db_bad_clust.data.db_connector import OracleConnector
from db_bad_clust.data.schema_extractor import SchemaExtractor
from db_bad_clust.features.structural_encoder import StructuralEncoder
from db_bad_clust.features.text_preprocessor import TextPreprocessor


def extract_schema(config: str):
    connector = OracleConnector(config_path=config)
    try:
        return SchemaExtractor(connector.connect()).extract_all()
    finally:
        connector.close()


def schema_from_pickle(path: str):
    """The `DatabaseSchema` an earlier run of this script stored under "schema".

    Lets a new block be added (or the encoder swapped) without the container
    up — the schema in the pickle is the one the extractor read, so the result
    is the same as extracting again.
    """
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    schema = data.get("schema")
    if schema is None:
        raise SystemExit(f"{path} carries no 'schema' key — rebuild it from Oracle first")
    return schema


def build(
    schema,
    model_name: str,
    include_table_comment: bool = True,
    semantic: str = "documents",
    mismatch: bool = False,
    text_prefix: str = "",
    conflict_mode: str = "hard",
) -> dict[str, object]:
    """Documents → embeddings, plus the structural blocks, in one column order.

    `semantic="documents"` puts the raw 384-dim document embedding in e_text.
    `semantic="anchors"` puts the compact anchor block there instead: one
    dimension per named concept plus the type-mismatch feature, which is
    dimensionally comparable to the structural block and readable per column.
    The anchors read the *name*, not the document — the document states the
    type, which would tell the encoder the answer it is being asked for.

    `mismatch=True` (only meaningful with semantic="documents"; a no-op with
    "anchors", which already carries the same scalar as its last dimension)
    appends the type-mismatch scalar as a second column of e_stat.

    Why it goes in e_stat and not as its own weighted block: the raw document
    embedding states the type in words — "tipo texto de longitud variable" —
    but a general-purpose sentence encoder does not reliably make the
    discordance between a name and its type a salient axis on its own. A
    supervised probe on the same features found exactly this: with the
    document alone, 12 of 27 wrong_data_types columns are predicted `clean`
    (F1-macro 0.4118); adding the mismatch scalar explicitly lifts that to
    0.4255. `FeatureBuilder` already z-scores each dimension before scaling
    the block, so a second column on a different natural scale (0-1 mismatch
    against log-length) needs no new normalisation logic — this is exactly
    the case `e_stat` (delta) was for.

    `text_prefix` is prepended to every document before encoding (semantic=
    "documents" only). Some encoders are trained with instruction prefixes and
    score worse without them — e.g. the intfloat/multilingual-e5-* family's own
    model card documents a "query: " prefix, including for clustering per its
    FAQ. Not applying it isn't a neutral no-op for those models, it's silently
    reproducing the wrong documented usage.

    `conflict_mode` is ConflictBlock's expectation: "hard" (default, one-hot on
    the closest family) or "soft" (softmax, reproduces candidato #6).
    """
    preprocessor = TextPreprocessor.for_documents()
    documents, keys = preprocessor.build_documents(
        schema, include_table_comment=include_table_comment
    )
    columns = [col for table in schema.tables for col in table.columns]
    names = [
        preprocessor.process(col.name, table.name)
        for table in schema.tables
        for col in table.columns
    ]

    from db_bad_clust.features.bert_embedder import BERTEmbedder
    from db_bad_clust.features.semantic_anchors import (
        TYPE_FAMILY_ANCHOR_VARIANTS,
        ConflictBlock,
        SemanticAnchors,
    )

    embedder = BERTEmbedder(model_name=model_name)

    if semantic == "anchors":
        e_text = SemanticAnchors(embedder).build_block(names, columns)
    else:
        prefixed = [text_prefix + d for d in documents] if text_prefix else documents
        e_text = embedder.encode(prefixed)

    encoder = StructuralEncoder()
    blocks = encoder.encode_all(columns)
    e_stat = blocks["statistical"]

    if mismatch and semantic != "anchors":
        anchor_mismatch = SemanticAnchors(embedder).type_mismatch(names, columns)
        e_stat = np.hstack([e_stat, anchor_mismatch])

    # The conflict block reads the bare name — no table, no document — because
    # the document states the type, which is the answer being asked for, and
    # the table name would put topic back into a block whose point is to have
    # none.
    bare_names = [preprocessor.process(col.name) for col in columns]
    e_conflict_variants = {
        wording: ConflictBlock(embedder, anchors=anchors, mode=conflict_mode).build(
            bare_names, columns
        )
        for wording, anchors in TYPE_FAMILY_ANCHOR_VARIANTS.items()
    }
    e_conflict = e_conflict_variants["orig"]

    return {
        "e_text": e_text,
        "e_type": blocks["data_types"],
        "e_rest": blocks["constraints"],
        "e_stat": e_stat,
        "e_conflict": e_conflict,
        "e_conflict_variants": e_conflict_variants,
        "conflict_mode": conflict_mode,
        "column_index": keys,
        "documents": documents,
        "include_table_comment": include_table_comment,
        "semantic": semantic,
        "mismatch": mismatch and semantic != "anchors",
        "schema": schema,
        "all_columns": columns,
        "table_names": [t.name for t in schema.tables],
        "model_name": model_name,
        "text_prefix": text_prefix,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="output/intermediate_docs.pkl")
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Hugging Face sentence encoder; the schema is Spanish, so keep it multilingual",
    )
    parser.add_argument(
        "--semantic",
        choices=("documents", "anchors"),
        default="documents",
        help="what goes in e_text: the raw document embedding, or the anchor block",
    )
    parser.add_argument(
        "--query-prefix",
        default="",
        help='text prepended to every document before encoding (documents mode only), '
        'e.g. "query: " for the intfloat/multilingual-e5-* family — see that '
        "family's model card; not optional for those models, not needed for others",
    )
    parser.add_argument(
        "--mismatch",
        action="store_true",
        help="append the anchor type-mismatch scalar as a second e_stat column "
        "(documents mode only — anchors mode already carries it)",
    )
    parser.add_argument(
        "--no-table-comment",
        action="store_true",
        help="leave the table comment out of every document (ablation)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print a few documents and exit, without loading the model",
    )
    parser.add_argument(
        "--conflict-mode",
        choices=("hard", "soft"),
        default="hard",
        help="expectation in the conflict block: hard (default) or soft (candidato #6)",
    )
    parser.add_argument(
        "--from-pickle",
        metavar="PICKLE",
        default=None,
        help="read the schema from an existing pickle instead of Oracle (no container needed)",
    )
    args = parser.parse_args()

    schema = schema_from_pickle(args.from_pickle) if args.from_pickle else extract_schema(args.config)
    print(f"Schema: {len(schema.tables)} tables, {schema.total_columns()} columns")

    if args.dry_run:
        documents, keys = TextPreprocessor.for_documents().build_documents(
            schema, include_table_comment=not args.no_table_comment
        )
        print(f"{len(documents)} documents\n")
        for key, doc in list(zip(keys, documents, strict=True))[:5]:
            print(f"  {key}\n    {doc}\n")
        return

    data = build(
        schema,
        args.model,
        include_table_comment=not args.no_table_comment,
        semantic=args.semantic,
        mismatch=args.mismatch,
        text_prefix=args.query_prefix,
        conflict_mode=args.conflict_mode,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as fh:
        pickle.dump(data, fh)

    e_text = data["e_text"]
    print(f"Embedded {len(data['documents'])} documents with {args.model}")
    print(
        f"e_text {np.shape(e_text)}, e_stat {np.shape(data['e_stat'])}, "
        f"e_conflict {np.shape(data['e_conflict'])} → {output}"
    )


if __name__ == "__main__":
    main()
