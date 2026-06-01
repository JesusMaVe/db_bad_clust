# Modulo 9: Recommender

## Proposito

Analizar los clusters generados y producir recomendaciones accionables para corregir anti-patrones de diseno de base de datos.

## Ubicacion

`scripts/recommender.py`

## Catalogo de Anti-Patrones

El recommender usa firmas conocidas para identificar problemas:

| Anti-Patron | Severidad | Deteccion |
|-------------|-----------|-----------|
| Tipo incorrecto | Alta | Nombre sugiere DATE/NUMBER pero tipo es VARCHAR |
| Nulos en PK | Alta | Columna "ID" permite NULL |
| Fecha como texto | Alta | Nombre contiene FECHA/DATE pero es VARCHAR |
| Numero como texto | Alta | Nombre contiene SALARIO/PRECIO pero es VARCHAR |
| Booleano inconsistente | Media | Nombre sugiere booleano pero es VARCHAR sin CHECK |
| VARCHAR sobredimensionado | Media | Longitud excesiva para el dominio |

## API

```python
from recommender import Recommender

recommender = Recommender()

recommendations = recommender.recommend(
    schema,           # DatabaseSchema
    labels,           # ndarray (N,) etiquetas de cluster
    column_table_map, # List[str] nombre de tabla por columna
)

# Formatear como texto
print(recommender.print_recommendations(recommendations))
```

## Output

Cada recomendacion incluye:
- cluster_id, total_columns, tables_involved
- dominant_type, pct_nullable, severity
- issues: lista de problemas detectados
- recommendations: lista de acciones correctivas
- columns: detalle de columnas en el cluster
