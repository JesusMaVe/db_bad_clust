# Módulo 1: Schema Extractor

## Propósito

Extrae el esquema completo de una base de datos Oracle: nombres de tablas,
columnas, tipos de dato, nulabilidad, restricciones (PK, FK, Unique),
e índices. Es el punto de entrada de la Fase 2.

## Ubicación

`scripts/schema_extractor.py`

## Dependencias

- `oracledb` — conexión a Oracle
- Solo depende del código existente (`OracleConnector`)

## Data Classes

```
DatabaseSchema
 ├── tables: List[TableMetadata]
 └── table_names() -> List[str]
     total_columns() -> int
     to_dict() -> dict

TableMetadata
 ├── name: str
 ├── columns: List[ColumnMetadata]
 ├── table_comment: Optional[str]
 └── row_count_approx: Optional[int]

ColumnMetadata
 ├── name: str
 ├── data_type: str
 ├── nullable: bool
 ├── data_length, data_precision, data_scale: Optional[int]
 ├── is_primary_key: bool
 ├── is_foreign_key: bool
 ├── is_unique: bool
 ├── is_indexed: bool
 ├── fk_references_table, fk_references_column, fk_name: Optional[str]
 ├── default_value: Optional[str]
 └── comments: Optional[str]
```

## API

```python
from schema_extractor import SchemaExtractor

extractor = SchemaExtractor(connection)
schema = extractor.extract_all()

# Iterar
for table in schema.tables:
    for col in table.columns:
        print(col.name, col.data_type, col.is_primary_key)
```

## Vistas del Diccionario de Datos Consultadas

| Vista Oracle               | Propósito                  |
|----------------------------|----------------------------|
| `user_tables`              | Listado de tablas          |
| `user_tab_columns`         | Columnas, tipos, nulos     |
| `user_constraints`         | Restricciones (P, R, U)    |
| `user_cons_columns`        | Columnas por constraint    |
| `user_ind_columns`         | Columnas indexadas         |

## Salida de Ejemplo

```
=== EMPLEADOS (7 cols) ===
  ID                  NUMBER        NULL
  NOMBRE_EMPLEADO     VARCHAR2      NULL
  FECHA_NACIMIENTO    VARCHAR2      NULL
  SALARIO             VARCHAR2      NULL
  DEPARTAMENTO        NUMBER        NULL
  EMAIL               DATE          NULL
  ACTIVO              VARCHAR2      NULL
```

## Notas

- Opera sobre el schema del usuario conectado (`user_*` vistas).
- No modifica la base de datos — solo lectura.
- Las columnas sin PK/FK/índices quedan con flags `False` (como en las
  tablas con anti-patrones de Phase 1).
