"""Tests for the command-line entry point. No database required."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from db_bad_clust.cli import build_parser, main
from db_bad_clust.data.schema_extractor import ColumnMetadata, DatabaseSchema, TableMetadata
from db_bad_clust.exceptions import BadDBError


def _schema() -> DatabaseSchema:
    return DatabaseSchema(
        tables=[
            TableMetadata(
                name="VENTAS",
                columns=[
                    ColumnMetadata(name="ID", data_type="NUMBER", nullable=False),
                    ColumnMetadata(
                        name="FECHA", data_type="VARCHAR2", nullable=True, data_length=20
                    ),
                ],
            )
        ]
    )


class TestParser:
    def test_requires_a_subcommand(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_audit_defaults_to_repo_config(self):
        args = build_parser().parse_args(["audit"])
        assert args.config == "config.yaml"
        assert args.sql is None

    def test_compare_defaults_need_no_database(self):
        args = build_parser().parse_args(["compare"])
        assert args.pickle == "output/intermediate_02.pkl"
        assert args.labels == "output/manual_labels.csv"
        assert args.sweep is False


class TestAudit:
    @patch("db_bad_clust.data.schema_extractor.SchemaExtractor")
    @patch("db_bad_clust.data.db_connector.OracleConnector")
    def test_reports_table_and_column_counts(self, connector, extractor, capsys):
        extractor.return_value.extract_all.return_value = _schema()
        assert main(["audit"]) == 0
        assert "1 tables, 2 columns" in capsys.readouterr().out

    @patch("db_bad_clust.data.schema_extractor.SchemaExtractor")
    @patch("db_bad_clust.data.db_connector.OracleConnector")
    def test_writes_corrective_sql_when_asked(self, connector, extractor, tmp_path, capsys):
        extractor.return_value.extract_all.return_value = _schema()
        target = tmp_path / "nested" / "fix.sql"
        assert main(["audit", "--sql", str(target)]) == 0
        assert target.exists()
        assert "db_bad_clust" in target.read_text()
        assert str(target) in capsys.readouterr().out

    @patch("db_bad_clust.data.schema_extractor.SchemaExtractor")
    @patch("db_bad_clust.data.db_connector.OracleConnector")
    def test_writes_json_findings_when_asked(self, connector, extractor, tmp_path):
        extractor.return_value.extract_all.return_value = _schema()
        target = tmp_path / "audit.json"
        assert main(["audit", "--json", str(target)]) == 0
        payload = json.loads(target.read_text())
        assert payload["tables"] == 1
        assert payload["columns"] == 2
        assert "column_issues" in payload["recommendations"]

    @patch("db_bad_clust.data.schema_extractor.SchemaExtractor")
    @patch("db_bad_clust.data.db_connector.OracleConnector")
    def test_closes_the_connection_even_when_extraction_fails(self, connector, extractor):
        instance = MagicMock()
        connector.return_value = instance
        extractor.return_value.extract_all.side_effect = BadDBError("boom")
        assert main(["audit"]) == 1
        instance.close.assert_called_once()


class TestErrorHandling:
    @patch("db_bad_clust.data.db_connector.OracleConnector")
    def test_database_errors_exit_nonzero_with_a_message(self, connector, capsys):
        connector.return_value.connect.side_effect = BadDBError("no listener")
        assert main(["audit"]) == 1
        assert "no listener" in capsys.readouterr().err

    def test_missing_input_file_reports_the_path(self, capsys):
        code = main(["compare", "--pickle", "does/not/exist.pkl"])
        assert code == 1
        assert "does/not/exist.pkl" in capsys.readouterr().err
