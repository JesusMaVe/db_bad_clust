# Módulo 2: Text Preprocessor

## Propósito

Transforma nombres técnicos de columnas (camelCase, snake_case, UPPER_CASE)
en texto limpio y semánticamente enriquecido para alimentar BERT.

## Ubicación

`scripts/text_preprocessor.py`

## Pipeline

```
Nombre original → Split camelCase → Split _ → Remove prefixes
→ Expand abbreviations → Lowercase → Contextualize → Salida
```

## Ejemplos

| Entrada                          | Salida                                  |
|----------------------------------|-----------------------------------------|
| `FECHA_NACIMIENTO` + `EMPLEADOS` | `empleados: fecha nacimiento`           |
| `ID_CLIENTE` + `CLIENTES`        | `clientes: identifier cliente`          |
| `isActive` + `USERS`             | `users: is active`                      |
| `XMLConfig` + `SETTINGS`         | `settings: xml configuration`           |
| `FK_PROVINCIA` + `CIUDADES`      | `ciudades: foreign key provincia`       |
| `TBL_DATOS` + `TBL_DATOS`        | `datos: datos`                          |
| `UUID_VALOR` + `TABLA_X`         | `tabla x: unique identifier valor`      |

## API

```python
from text_preprocessor import TextPreprocessor, preprocess

prep = TextPreprocessor()

# Individual
text = prep.process("FECHA_NACIMIENTO", table_name="EMPLEADOS")

# Batch
texts = prep.process_batch(["ID", "NOMBRE"], table_name="EMPLEADOS")

# Función directa
text = preprocess("isActive", "USERS")
```

## Abreviaturas Soportadas (parcial)

| Abreviatura | Expansión          |
|-------------|--------------------|
| id          | identifier         |
| pk          | primary key        |
| fk          | foreign key        |
| num / no    | number             |
| qty         | quantity           |
| desc        | description        |
| dept        | department         |
| org         | organization       |
| emp         | employee           |
| prod        | product            |
| cat         | category           |
| uuid / guid | unique identifier  |
| dt          | date               |
| ts          | timestamp          |

## Notas

- Diseñado para `bert-base-multilingual-cased` — preserva palabras en español.
- Los nombres crípticos como `C1`, `C2` se mantienen (son anti-patrones válidos).
- El contexto de tabla se antepone con `": "` para que BERT diferencie
  columnas con el mismo nombre en tablas distintas.
