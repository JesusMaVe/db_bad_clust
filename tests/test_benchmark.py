"""
Tests for benchmark.py.

All tests are mock-based and do not require a running Oracle database.
"""

from __future__ import annotations

from db_bad_clust.generation.benchmark import run_benchmark
from db_bad_clust.generation.schema_generator import SchemaGeneratorConfig, SyntheticSchemaGenerator


class TestBenchmark:
    def test_benchmark_runs_and_produces_report(self):
        report = run_benchmark(50, seed=1, density=1.0)
        assert report.num_tables == 50
        assert report.num_columns > 0
        assert report.generation_time_ms >= 0
        assert report.detection_time_ms >= 0
        assert "clean" in report.column_metrics
        assert "fk_without_index" in report.report_metrics

    def test_benchmark_perfect_on_handcrafted_schema(self):
        # A single FK-without-index table should be detected with perfect metrics.
        generator = SyntheticSchemaGenerator(seed=7)
        tables, manifest = generator.generate(
            SchemaGeneratorConfig(num_tables=10, seed=7, anti_pattern_density=0.0)
        )
        # Force at least one FK-without-index table by injecting a template.
        from db_bad_clust.generation.schema_generator import _fk_without_index_table

        extra = _fk_without_index_table(generator.rng, 999)
        tables.append(extra)
        manifest["tables"].append(
            {
                "name": extra.name,
                "anti_patterns": [],
                "report_flags": {"fk_without_index": True},
                "columns": [
                    {"name": c, "data_type": t, "anti_pattern": "clean"}
                    for c, t in extra.columns.items()
                ],
                "has_redundant_indexes": False,
            }
        )
        schema = SyntheticSchemaGenerator.to_database_schema(tables)

        from db_bad_clust.generation.benchmark import (
            _build_ground_truth,
            _collect_detections,
            _evaluate_report_level,
        )

        detections = _collect_detections(schema)
        _, table_truth, _ = _build_ground_truth(tables, manifest)
        metrics = _evaluate_report_level(table_truth, {}, detections)
        assert metrics["fk_without_index"]["recall"] == 1.0
        assert metrics["fk_without_index"]["precision"] == 1.0

    def test_column_level_ground_truth_matches_rule_engine(self):
        # With the aligned manifest, classify() should match ground truth perfectly.
        report = run_benchmark(100, seed=42, density=0.4)
        for label, m in report.column_metrics.items():
            if m["support"] == 0:
                continue
            assert m["f1"] == 1.0, f"{label} is not perfectly aligned"
