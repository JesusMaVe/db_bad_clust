# Modulo 2: Text Preprocessor

## Proposito

Transforma nombres tecnicos de columnas (camelCase, snake_case, UPPER_CASE)
en texto limpio y semanticamente enriquecido para alimentar BERT.

## Ubicacion

`scripts/text_preprocessor.py`

## Pipeline

```
Nombre original -> Split camelCase -> Split _ -> Remove prefixes
-> Expand abbreviations -> Lowercase -> Contextualize -> Salida
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

# Funcion directa
text = preprocess("isActive", "USERS")
```

## Abreviaturas Soportadas (parcial)

| Abreviatura | Expansion          |
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

- Disenyado para `bert-base-multilingual-cased`  preserva palabras en espanyol.
- Los nombres cripticos como `C1`, `C2` se mantienen (son anti-patrones validos).
- El contexto de tabla se antepone con `": "` para que BERT diferencie
  columnas con el mismo nombre en tablas distintas.
