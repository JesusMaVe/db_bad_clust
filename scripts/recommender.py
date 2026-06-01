"""
recommender.py — Recommendation generation for fixing anti-patterns

Purpose:
  Analyze generated clusters and produce actionable recommendations
  to fix database design anti-patterns.

  Per cluster:
    1. Identify dominant characteristics (types, nullability, etc.)
    2. Compare against known best practices
    3. Generate natural language recommendation
    4. Prioritize by severity

Usage:
  recommender = Recommender()
  recommendations = recommender.recommend(schema, labels, column_table_map)
  recommender.print_recommendations(recommendations)
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from schema_extractor import ColumnMetadata, DatabaseSchema

logger = logging.getLogger(__name__)

# Catalog of known anti-pattern descriptions
ANTI_PATTERN_SIGNATURES = [
    {
        "name": "Tipo incorrecto",
        "description": "Columna con tipo de dato inapropiado para su contenido semantico",
        "severity": "alta",
        "fix": "Usar el tipo de dato nativo correspondiente (DATE para fechas, NUMBER para numeros)",
    },
    {
        "name": "Nulos en PK",
        "description": "Columna nombrada como identificador pero permite nulos",
        "severity": "alta",
        "fix": "Agregar constraint NOT NULL y considerar una clave primaria real",
    },
    {
        "name": "VARCHAR sobredimensionado",
        "description": "Columna VARCHAR con longitud excesiva para el dominio de datos",
        "severity": "media",
        "fix": "Reducir la longitud al maximo real del dominio, o usar CLOB si es necesario",
    },
    {
        "name": "Fecha como texto",
        "description": "Fechas almacenadas en VARCHAR en vez de DATE o TIMESTAMP",
        "severity": "alta",
        "fix": "Migrar a columna DATE o TIMESTAMP con validacion de formato",
    },
    {
        "name": "Numero como texto",
        "description": "Valores numericos almacenados como VARCHAR (impiden calculos)",
        "severity": "alta",
        "fix": "Convertir a NUMBER con precision y escala adecuadas",
    },
    {
        "name": "Sin indice",
        "description": "Columna usada en JOINs o WHERE sin indice",
        "severity": "media",
        "fix": "Crear indice B-tree en columnas de FK y filtros frecuentes",
    },
    {
        "name": "Booleano inconsistente",
        "description": "Columna booleana con multiples convenciones (S/N, 1/0, True/False, Si/No)",
        "severity": "media",
        "fix": "Unificar a CHAR(1) con CHECK IN ('S','N') o NUMBER(1) con CHECK IN (0,1)",
    },
]


class Recommender:
    """
    Generates correction recommendations based on clusters and metadata.
    """

    def __init__(self, signatures: list[dict[str, Any]] | None = None) -> None:
        self.signatures = signatures or ANTI_PATTERN_SIGNATURES

    def recommend(
        self,
        schema: DatabaseSchema,
        labels: np.ndarray,
        column_table_map: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate recommendations for each cluster.

        Args:
            schema: DatabaseSchema extracted from Oracle.
            labels: Cluster labels per column.
            column_table_map: Column -> table name mapping.

        Returns:
            List of structured recommendations.
        """
        all_columns = []
        for table in schema.tables:
            for col in table.columns:
                all_columns.append({"column": col, "table": table.name})

        if column_table_map is None:
            column_table_map = [c["table"] for c in all_columns]

        clusters = defaultdict(list)
        for idx, (label, col_info) in enumerate(zip(labels, all_columns)):
            clusters[int(label)].append(
                {
                    "index": idx,
                    "table": col_info["table"],
                    "column": col_info["column"],
                }
            )

        recommendations = []
        for cluster_id, members in sorted(clusters.items()):
            if cluster_id == -1:
                continue

            cols_in_cluster = [m["column"] for m in members]
            tables_in_cluster = list(set(m["table"] for m in members))
            types = Counter(c.data_type for c in cols_in_cluster)
            nullable_count = sum(1 for c in cols_in_cluster if c.nullable)

            cluster_recs = self._analyze_cluster(
                cluster_id, members, cols_in_cluster, types, nullable_count, tables_in_cluster
            )
            recommendations.append(cluster_recs)

        return recommendations

    def _analyze_cluster(
        self,
        cluster_id: int,
        members: list[dict[str, Any]],
        cols: list[ColumnMetadata],
        types: Counter,
        nullable_count: int,
        tables: list[str],
    ) -> dict[str, Any]:
        """Analyze an individual cluster and generate recommendations."""
        total = len(members)
        pct_nullable = nullable_count / total * 100 if total else 0
        dominant_type = types.most_common(1)[0][0] if types else "unknown"

        issues = []
        fixes = []

        if pct_nullable > 50:
            issues.append(f"Mayoria de columnas permite nulos ({pct_nullable:.0f}%)")
            fixes.append(
                "Evaluar si los nulos son semanticamente validos o indican datos faltantes"
            )

        if dominant_type.upper().startswith("VARCHAR"):
            # Check if any column looks like a date or number stored as text
            date_keywords = {"FECHA", "DATE", "BIRTH", "ALTA", "CREACION", "ACTUALIZACION"}
            num_keywords = {
                "SALARIO",
                "PRECIO",
                "COSTO",
                "MONTO",
                "TOTAL",
                "IMPORTE",
                "CANTIDAD",
                "SUELDO",
            }
            boolean_keywords = {"ACTIVO", "ACTIVE", "FLAG", "BOOL", "VIGENTE", "ESTADO"}

            date_cols = [
                m for m in members if any(k in m["column"].name.upper() for k in date_keywords)
            ]
            num_cols = [
                m for m in members if any(k in m["column"].name.upper() for k in num_keywords)
            ]
            bool_cols = [
                m for m in members if any(k in m["column"].name.upper() for k in boolean_keywords)
            ]

            for col_info in date_cols:
                issues.append(
                    f"'{col_info['column'].name}' parece fecha pero es {col_info['column'].data_type}"
                )
                fixes.append("Fecha como texto: migrar a DATE")

            for col_info in num_cols:
                issues.append(
                    f"'{col_info['column'].name}' parece numero pero es {col_info['column'].data_type}"
                )
                fixes.append("Numero como texto: migrar a NUMBER")

            for col_info in bool_cols:
                issues.append(
                    f"'{col_info['column'].name}' parece booleano pero es {col_info['column'].data_type}"
                )
                fixes.append("Booleano como texto: unificar a CHAR(1) con CHECK")

        if not issues:
            issues.append(f"Tipo dominante: {dominant_type}")
            fixes.append("Monitorear consistencia del cluster")

        severity = (
            "alta"
            if any("Fecha como texto" in f or "Numero como texto" in f for f in fixes)
            else "media"
        )

        return {
            "cluster_id": int(cluster_id),
            "total_columns": total,
            "tables_involved": sorted(set(tables)),
            "dominant_type": dominant_type,
            "pct_nullable": round(pct_nullable, 1),
            "severity": severity,
            "issues": issues,
            "recommendations": fixes,
            "columns": [
                {"table": m["table"], "name": m["column"].name, "type": m["column"].data_type}
                for m in members
            ],
        }

    @staticmethod
    def print_recommendations(recommendations: list[dict[str, Any]]) -> str:
        """Format recommendations as text."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("RECOMMENDATIONS PER CLUSTER")
        lines.append("=" * 60)

        for rec in recommendations:
            lines.append(
                f"\nCluster #{rec['cluster_id']} "
                f"[{rec['severity'].upper()}] "
                f"({rec['total_columns']} columns, "
                f"{len(rec['tables_involved'])} tables)"
            )
            lines.append(f"  Dominant type: {rec['dominant_type']}")
            lines.append(f"  Tables: {', '.join(rec['tables_involved'][:5])}")
            if len(rec["tables_involved"]) > 5:
                lines.append(f"    ... and {len(rec['tables_involved']) - 5} more")

            for issue in rec["issues"]:
                lines.append(f"  - {issue}")

            for fix in rec["recommendations"]:
                lines.append(f"    -> {fix}")

        return "\n".join(lines)
