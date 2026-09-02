# db_bad_clust — detección no supervisada de anti-patrones de esquema Oracle

Agrupa las columnas de una base Oracle en anti-patrones de diseño **sin etiquetas**, usando
embeddings de oraciones (BERT multilingüe) sobre una descripción en lenguaje natural de cada
columna, fusionados con características estructurales.

> **Dos ramas.** `main` es la investigación (esta). La rama **`rule-engine`** contiene el motor
> de reglas heurístico y el producto de auditoría Oracle (`cli audit`, SQL correctivo). Son
> trabajos distintos con objetivos distintos; no se mezclan.

## La idea

Un anti-patrón como `wrong_data_types` es una **discordancia**: la columna se llama
`FECHA_INGRESO` pero está declarada `VARCHAR2(20)`. Detectarla requiere leer las dos mitades a
la vez. El pipeline le da al encoder una frase que contiene ambas:

```
tabla empleados, columna fecha ingreso, tipo texto de longitud variable
de hasta 20 caracteres, admite nulos. La tabla contiene: ...
```

y agrupa los vectores resultantes. La verdad de terreno son 243 columnas etiquetadas a mano
(`output/manual_labels.csv`), que el pipeline nunca ve.

## Resultados

Sobre las 153 columnas que quedan al excluir las dos tablas gigantes — `giant_table` es
propiedad *de la tabla*, no de la columna, así que dejarla dentro premia la fuga de identidad
de tabla en vez de la separación de anti-patrones. α = 1.00:

| representación                |     ARI |    NMI |    AMI | Accuracy | F1-macro |
| ----------------------------- | ------: | -----: | -----: | -------: | -------: |
| solo estructura (α=0)         |  0.0657 | 0.2432 | 0.1212 |   0.6078 |   0.2002 |
| embedding del nombre (previo) | -0.0538 | 0.2043 | 0.0747 |   0.5425 |   0.1346 |
| **documento de columna**      |  0.0802 | 0.3788 | 0.2553 |   0.6797 |   0.3781 |

Sobre el corpus completo (243 columnas), el documento sube AMI de 0.3619 a **0.4818**, accuracy
de 0.7119 a **0.7984** y F1-macro de 0.2670 a **0.4622**.

*Accuracy y F1 de un clustering usan nombrado por voto mayoritario, que consulta la verdad de
terreno: son cota superior, no marca alcanzada.*

### Tres hallazgos que cambiaron la conclusión

1. **El peso α no ponderaba nada.** La normalización z-score por dimensión deja la varianza
   total de un bloque igual a su número de dimensiones (384 contra 18). Con α nominal 0.15 los
   embeddings se quedaban el **89%** del espacio. La conclusión previa de que "BERT estorba"
   comparaba *sin embeddings* contra *embeddings y casi nada más*.

2. **El baseline estructural no distingue las columnas.** Con α=0, 243 columnas colapsan en
   **24 vectores distintos** (el mayor grupo tiene 108). Dos columnas con el mismo vector no
   pueden recibir etiquetas distintas de ningún algoritmo. Su ARI tampoco es reproducible:
   **0.1877 en float32 contra 0.5031 en float64**.

3. **El embedding leía un sintagma sin tipo.** `"empleados: fecha nacimiento"` no permite
   distinguir un DATE de un VARCHAR2. Con el nombre solo, el embedding daba ARI **negativo**.

## Uso

```bash
docker compose up -d
.venv/bin/python -m pip install -e ".[dev]"

.venv/bin/python -m db_bad_clust.cli experiment --sweep
.venv/bin/python -m db_bad_clust.cli experiment --without-giants \
    --ablation output/intermediate_02.pkl output/intermediate_docs.pkl
```

Reconstruir los embeddings desde Oracle:

```bash
.venv/bin/python scripts/apply_comments.py                  # una vez: comentarios en la BD
.venv/bin/python scripts/build_embeddings.py --dry-run      # ver los documentos
.venv/bin/python scripts/build_embeddings.py --output output/intermediate_docs.pkl
```

```bash
.venv/bin/python -m pytest tests/ -v       # 372 tests, sin BD ni descarga del modelo
```

`AGENTS.md` tiene el detalle completo: invariantes, ablaciones y limitaciones medidas.

## Limitaciones

- Un corpus: 23 tablas, 243 columnas, una sola base sintética en español. Los números no se
  transfieren sin más a otro esquema.
- Clases muy desbalanceadas: `giant_table` es el 37% del corpus y `self_referencing` tiene una
  sola columna.
- `impossible_data` no es detectable desde el metadata en esta base: no hay constraints R y las
  claves padre tienen duplicados y nulos, así que las FK no pueden ni crearse.
- Las anclas semánticas (12 dimensiones interpretables) **pierden** contra las 384 crudas como
  representación de agrupamiento. Se conservan como explicación por columna, no como mejora.
