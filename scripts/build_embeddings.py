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

Usage:
    .venv/bin/python scripts/build_embeddings.py --output output/intermediate_docs.pkl
    .venv/bin/python scripts/build_embeddings.py --dry-run          # print 5 documents, no model
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


def build(
    schema,
    model_name: str,
    include_table_comment: bool = True,
    semantic: str = "documents",
) -> dict[str, object]:
    """Documents → embeddings, plus the structural blocks, in one column order.

    `semantic="documents"` puts the raw 384-dim document embedding in e_text.
    `semantic="anchors"` puts the compact anchor block there instead: one
    dimension per named concept plus the type-mismatch feature, which is
    dimensionally comparable to the structural block and readable per column.
    The anchors read the *name*, not the document — the document states the
    type, which would tell the encoder the answer it is being asked for.
    """
    preprocessor = TextPreprocessor.for_documents()
    documents, keys = preprocessor.build_documents(
        schema, include_table_comment=include_table_comment
    )
    columns = [col for table in schema.tables for col in table.columns]

    from db_bad_clust.features.bert_embedder import BERTEmbedder

    embedder = BERTEmbedder(model_name=model_name)

    if semantic == "anchors":
        from db_bad_clust.features.semantic_anchors import SemanticAnchors

        names = [
            preprocessor.process(col.name, table.name)
            for table in schema.tables
            for col in table.columns
        ]
        e_text = SemanticAnchors(embedder).build_block(names, columns)
    else:
        e_text = embedder.encode(documents)

    encoder = StructuralEncoder()
    blocks = encoder.encode_all(columns)

    return {
        "e_text": e_text,
        "e_type": blocks["data_types"],
        "e_rest": blocks["constraints"],
        "e_stat": blocks["statistical"],
        "column_index": keys,
        "documents": documents,
        "include_table_comment": include_table_comment,
        "semantic": semantic,
        "schema": schema,
        "all_columns": columns,
        "table_names": [t.name for t in schema.tables],
        "model_name": model_name,
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
        "--no-table-comment",
        action="store_true",
        help="leave the table comment out of every document (ablation)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print a few documents and exit, without loading the model",
    )
    args = parser.parse_args()

    schema = extract_schema(args.config)
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
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as fh:
        pickle.dump(data, fh)

    e_text = data["e_text"]
    print(f"Embedded {len(data['documents'])} documents with {args.model}")
    print(f"e_text {np.shape(e_text)} → {output}")


if __name__ == "__main__":
    main()
