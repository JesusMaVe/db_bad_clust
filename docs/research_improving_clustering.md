# Investigación: técnicas para mejorar el resultado central de clustering no supervisado

**Fecha:** 2026-09-17
**Método:** revisión de fuentes primarias (documentación oficial de sentence-transformers/Hugging Face, MTEB/MMTEB, `hdbscan`/scikit-learn, UMAP, papers en arXiv sobre SBERT/SetFit/Deep Sets/multi-view clustering; búsquedas dirigidas en arXiv/ACM para detección de anti-patrones de esquema).
**Alcance:** (1) modelos de embeddings de oraciones multilingües alternativos a `paraphrase-multilingual-MiniLM-L12-v2`; (2) guía oficial de parámetros de HDBSCAN; (3) PCA vs UMAP antes de clustering basado en densidad; (4) alternativas a la concatenación ponderada para fusionar semántica y estructura; (5) precedente en la literatura para añadir señal agregada a nivel de tabla sin reglas; (6) trabajo previo en detección de anti-patrones de esquema con embeddings/ML.

## Qué cuenta como "mejora" aquí

Esta investigación respeta tres restricciones del pipeline, explícitas en `CLAUDE.md`/`AGENTS.md` y en el encargo:

1. **El pipeline debe seguir siendo no supervisado en inferencia.** Las 243 etiquetas de `output/manual_labels.csv` solo puntúan una configuración después del hecho (la metodología de barrido ya establecida en `evaluation/experiments.py`); nunca son entrada directa de una clasificación por columna durante el clustering. Cualquier técnica que necesite etiquetas por columna para *correr* queda fuera de alcance — para eso está el clasificador supervisado de `evaluation/health_report.py`, una etapa aparte que no se debe confundir con esta.
2. **No reintroducir el motor de reglas de `rule-engine`.** Cualquier señal a nivel de tabla debe ser una feature numérica/estadística, no un if/else.
3. **Debe correr con lo que ya es dependencia** (`scikit-learn>=1.3.0` → instalado 1.9.1; `umap-learn>=0.5.0` → instalado 0.5.12; `hdbscan>=0.8.33`; `transformers>=4.30.0` + `torch>=2.0.0`) **o con una dependencia nueva justificada**. Nota de implementación importante: `pyproject.toml` no lista `sentence-transformers` como dependencia y no está instalado en el venv del repo (`import sentence_transformers` falla); `features/bert_embedder.py` usa `transformers.AutoModel`/`AutoTokenizer` directamente con mean pooling manual + normalización L2 escritos a mano — exactamente el mismo cómputo que la librería `sentence-transformers` hace internamente. Por eso cualquier checkpoint de Hugging Face compatible con `AutoModel`/`AutoTokenizer` (que incluye todos los modelos `sentence-transformers/*` e `intfloat/*` de la sección siguiente) se puede probar cambiando solo `model_name` en `BERTEmbedder`, sin añadir ninguna dependencia nueva.

---

## Tema 1: Modelos de embeddings de oraciones multilingües

### 1.1 Panorama: no hay un modelo universal, y el benchmark relevante es MTEB/MMTEB

- MTEB evalúa embeddings en 8 categorías de tarea (incluida clustering) sobre 58 datasets y 112 idiomas, precisamente porque *"los embeddings de texto se evalúan comúnmente en un pequeño conjunto de datasets de una sola tarea"* que no cubre otros usos como clustering. Su hallazgo central: *"ningún método particular de embedding de texto domina en todas las tareas"*. Fuente: [MTEB: Massive Text Embedding Benchmark (arXiv:2210.07316)](https://arxiv.org/abs/2210.07316)
- La sucesora multilingüe, MMTEB, cubre *"más de 500 tareas de evaluación con control de calidad en más de 250 idiomas"*. Su hallazgo relevante para elegir un modelo pequeño: *"el mejor modelo públicamente disponible es multilingual-e5-large-instruct, con solo 560 millones de parámetros"* — es decir, ni siquiera el mejor modelo público necesita ser gigante, pero 560M es ~5× más grande que el modelo actual del pipeline (ver 1.2). Fuente: [MMTEB: Massive Multilingual Text Embedding Benchmark (arXiv:2502.13595)](https://arxiv.org/abs/2502.13595)
- Consecuencia práctica para este pipeline: no existe una fuente primaria que diga "usa el modelo X para clustering de textos cortos en español" — la elección debe ser empírica (correr el mismo `evaluation/experiments.py --sweep` con cada modelo candidato), pero el listado de candidatos sí está soportado por fuentes primarias (model cards oficiales) en 1.2.

### 1.2 Candidatos concretos, compatibles con la API ya usada (`AutoModel`/`AutoTokenizer`, mean pooling)

| modelo | dim | idiomas | notas de la fuente primaria |
| --- | ---: | --- | --- |
| `paraphrase-multilingual-MiniLM-L12-v2` (actual) | 384 | 50+ | *"puede usarse para tareas como clustering o búsqueda semántica"*; max_seq_length 128; ~0.44 GB en disco. Fuente: [model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) |
| `intfloat/multilingual-e5-small` | 384 | ~100 | inicializado desde `microsoft/Multilingual-MiniLM-L12-H384` — **mismo backbone/clase de tamaño (12 capas, 384-dim) que el modelo actual**, pero continuado con entrenamiento contrastivo multilingüe en vez de solo paráfrasis. Fuente: [model card intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) |
| `intfloat/multilingual-e5-base` | 768 | 50+ (tabla MTEB con `es` incluido) | inicializado desde XLM-RoBERTa-base; reporta en su tarjeta scores de MTEB clustering (v_measure) en ArxivClusteringP2P=40.28, RedditClustering=42.41, superiores a los de e5-small (39.22 y 39.13 respectivamente) en las mismas tareas — más caro, algo mejor en clustering según su propio card. Fuente: [model card intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | 50+ | ~0.3B parámetros (XLM-RoBERTa-base), *"puede usarse para tareas como clustering o búsqueda semántica"* — mismo propósito que el MiniLM actual pero con backbone más grande. Fuente: [model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) |
| `BAAI/bge-m3` | 1024 | 100+ | soporta recuperación densa/dispersa/multi-vector simultáneamente, secuencias hasta 8192 tokens, funciona con `SentenceTransformer(...)` estándar (por tanto también con `AutoModel`) sin librería adicional obligatoria — pero está optimizado para documentos largos y recuperación, no específicamente para clustering de frases cortas. Fuente: [model card BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) |
| `sentence-transformers/LaBSE` | 768 | 109 | explícitamente **no recomendado** para este caso: *"funciona bien para encontrar pares de traducción en múltiples idiomas"* pero *"funciona peor evaluando la similitud de pares de oraciones que no son traducciones entre sí"* — el corpus no tiene pares de traducción, así que LaBSE es la peor apuesta de esta tabla pese a su cobertura de idiomas. Fuente: [sbert.net — Pretrained Models](https://sbert.net/docs/sentence_transformer/pretrained_models.html) |

### 1.3 Detalle de implementación que aplica a los modelos `e5-*`

- El model card de `multilingual-e5-base` documenta el uso de prefijos `"query: "` / `"passage: "` antes de cada texto, y su FAQ indica usar `"query: "` incluso para clustering — un detalle que hay que replicar en `TextPreprocessor`/`BERTEmbedder` si se prueba esta familia, no es opcional para reproducir el comportamiento documentado. Fuente: [model card intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)

---

## Tema 2: Guía oficial de HDBSCAN para corpus pequeños y tolerantes a ruido

### 2.1 Los parámetros que de verdad importan (documentación original de `hdbscan`)

- `min_cluster_size` es *"el parámetro principal que afecta al clustering resultante"*; debe fijarse al *"agrupamiento más pequeño que se quiera considerar un cluster"*. El pipeline actual lo fija a 5. Fuente: [hdbscan docs — Parameter Selection for HDBSCAN*](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html)
- `min_samples` controla *"qué tan conservador se quiere el clustering"*: valores más altos declaran más puntos como ruido y restringen los clusters a regiones más densas; **por defecto es igual a `min_cluster_size` si no se especifica**. El pipeline actual lo fija explícitamente a 2, **por debajo** de `min_cluster_size=5` — es decir, es *menos* conservador que el default documentado, no más. Esto no está mal per se, pero es una desviación deliberada del comportamiento por defecto que vale la pena barrer explícitamente (2, 3, 5) en vez de dar por sentado que 2 es la mejor elección. Fuente: [hdbscan docs — Parameter Selection](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html)
- `cluster_selection_method` tiene dos valores: `'eom'` (Excess of Mass, **default**, y el que usa el pipeline al no especificarlo) tiende a *"uno o dos clusters grandes más algunos pequeños extra"*; `'leaf'` *"seleccionará los nodos hoja del árbol, produciendo muchos clusters pequeños y homogéneos"` para un resultado más fino. Dado que el corpus objetivo tiene 10 etiquetas de anti-patrón y hoy produce k≈16-19 con `eom`, `'leaf'` es una alternativa de un solo parámetro, nunca probada en el repo, directamente motivada por esta guía. Fuente: [hdbscan docs — Parameter Selection](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html)
- `cluster_selection_epsilon` (default 0.0, no usado hoy) fusiona clusters separados por menos de esa distancia — *"fija el valor a 0.5 si no se quiere separar clusters a menos de 0.5 unidades"*. Útil si `'leaf'` produce demasiados micro-clusters. Fuente: [hdbscan docs — Parameter Selection](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html)
- `alpha` (escala la distancia en el single linkage robusto) se deja explícitamente para el final: la documentación recomienda ajustar primero `min_samples`/`cluster_selection_epsilon` y *"solo explorar alpha como último recurso"*. No es prioritario tocarlo aquí. Fuente: [hdbscan docs — Parameter Selection](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html)

### 2.2 ¿Aporta algo el HDBSCAN integrado en scikit-learn?

- `sklearn.cluster.HDBSCAN` (disponible desde 1.3, y el repo ya tiene 1.9.1 instalado) añade `algorithm` (`auto`/`brute`/`kd_tree`/`ball_tree`), `store_centers` (`"centroid"`/`"medoid"`/`"both"`) y un método `dbscan_clustering(cut_distance, min_cluster_size)` para extraer un corte DBSCAN* equivalente a una distancia concreta. Fuente: [scikit-learn — HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)
- **Gotcha documentado si se migra alguna vez de la librería `hdbscan` a `sklearn.cluster.HDBSCAN`:** la semántica de `min_samples` difiere en un punto — la versión de scikit-learn *"incluye el punto mismo"*, la de `scikit-learn-contrib/hdbscan` no, así que hay que sumar 1 al valor para reproducir el mismo resultado entre ambas. El pipeline actual usa la librería original (`import hdbscan`), así que esto solo importa si se prueba `sklearn.cluster.HDBSCAN` como alternativa. Fuente: [scikit-learn — HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)

---

## Tema 3: PCA vs UMAP antes de HDBSCAN

### 3.1 La combinación UMAP→HDBSCAN es explícitamente recomendada por la documentación de UMAP, con caveats igual de explícitos

- La página oficial "Using UMAP for Clustering" recomienda, para clustering (no para visualización), **subir `n_neighbors`** (ejemplo dado: de 15 a 30, porque *"valores pequeños se enfocan en estructura muy local y son propensos a producir estructura de cluster granular"*) y **bajar `min_dist` a 0.0** (*"queremos empaquetar puntos densamente... esto ayuda, además de crear separaciones más limpias entre clusters"*). Fuente: [umap-learn docs — Using UMAP for Clustering](https://umap-learn.readthedocs.io/en/latest/clustering.html)
- La misma página recomienda K-Means **en contra** para datos reducidos con UMAP porque *"UMAP no necesariamente produce clusters esféricos limpios"*, y recomienda HDBSCAN en su lugar porque puede *"identificar los componentes conectados que UMAP preserva bien"*. Esto es coherente con lo que el pipeline ya hace (HDBSCAN, no KMeans, en el clustering final). Fuente: [umap-learn docs — Using UMAP for Clustering](https://umap-learn.readthedocs.io/en/latest/clustering.html)
- **Caveat que hay que tomar en serio porque HDBSCAN es exactamente un algoritmo basado en densidad:** la propia documentación dice literalmente que *"UMAP, como t-SNE, no preserva completamente la densidad"* y que *"puede crear falsos desgarros (tears) en clusters, resultando en un clustering más fino de lo que realmente existe en los datos"*. Los autores describen el enfoque UMAP+HDBSCAN como *"algo controvertido, y debe intentarse con cuidado"*, recomendando explorar y validar los clusters resultantes cuando sea posible. Fuente: [umap-learn docs — Using UMAP for Clustering](https://umap-learn.readthedocs.io/en/latest/clustering.html)
- El FAQ oficial confirma lo mismo con otras palabras: *"UMAP, con su supuesto de densidad uniforme, no preserva bien la densidad"*, aunque sí es buena *"contrayendo componentes conectados de la variedad (manifold)"*, y advierte que **no cualquier algoritmo de clustering** debe aplicarse directamente sobre la salida de UMAP — la elección de algoritmo importa. Fuente: [umap-learn docs — FAQ](https://umap-learn.readthedocs.io/en/latest/faq.html)
- El paper original de UMAP (McInnes, Healy, Melville) respalda la motivación general: se presenta como *"competitivo con t-SNE en calidad de visualización, y presumiblemente preserva más estructura global, con mejor tiempo de ejecución"*, y sin restricciones computacionales sobre la dimensión de salida (a diferencia de t-SNE, que en la práctica se limita a 2-3D). Fuente: [UMAP: Uniform Manifold Approximation and Projection (arXiv:1802.03426)](https://arxiv.org/abs/1802.03426)

### 3.2 Aplicación a este pipeline

- `clustering/dimensionality_reducer.py` ya soporta `method="umap"` con defaults razonables (`n_neighbors=15, min_dist=0.1, metric="euclidean"`), pero `evaluation/experiments.py::_cluster` tiene **hardcodeado** `reducer="pca", n_components=20` — probar UMAP es tan barato como parametrizar esa llamada; `umap-learn` ya es dependencia instalada (0.5.12).
- Extrapolación propia, no textual de la fuente: el corpus sin las tablas gigantes tiene 153 filas. La recomendación de la documentación de subir `n_neighbors` a 30 para clustering fue dada sobre un dataset de decenas de miles de puntos (MNIST); 30 vecinos sobre 153 filas es ~20% del corpus, un radio de vecindad mucho más agresivo en relación al tamaño de la muestra que en el ejemplo de la documentación. Esto no está en ninguna fuente citada — es una razón concreta para barrer `n_neighbors` bajo (5–15) en vez de copiar literalmente el 30 recomendado para datasets grandes.

---

## Tema 4: Alternativas a la concatenación ponderada para fusionar semántica y estructura

### 4.1 Lo que hace hoy el pipeline es "fusión temprana" (early/feature-level fusion) en el vocabulario de la literatura de multi-view clustering

- `FeatureBuilder` normaliza cada bloque a varianza total unitaria y concatena con pesos α/β/γ/δ — es el patrón más simple de fusión a nivel de features, documentado en la literatura de multi-view clustering como una de varias familias de estrategias.
- El survey de referencia en el área revisa *"estrategias comunes para combinar múltiples vistas"* y propone una taxonomía de enfoques de multi-view clustering (MVC), relacionándolos con áreas vecinas como *"representación multi-vista, ensemble clustering, clustering multi-tarea, y aprendizaje supervisado/semi-supervisado multi-vista"* — confirma que la concatenación ponderada de bloques (lo que hace `FeatureBuilder`) es solo una de varias familias documentadas, no la única ni la más sofisticada. Fuente: [A Survey on Multi-View Clustering (arXiv:1712.06246)](https://arxiv.org/abs/1712.06246)

### 4.2 Fusión tardía (late fusion / consensus a nivel de partición) como alternativa concreta y sin nueva dependencia

- Un enfoque documentado de "late fusion multi-view clustering" agrupa **cada vista por separado** (p. ej., un clustering solo con el documento embebido y otro solo con las features estructurales) y luego fusiona **las particiones resultantes**, no los vectores de features — el paper propone maximizar la alineación entre una partición de consenso y las particiones base ponderadas, evitando el costo computacional de fusionar matrices de similitud completas. Fuente: [Late Fusion Multi-view Clustering via Global and Local Alignment Maximization (arXiv:2208.01198)](https://arxiv.org/abs/2208.01198)
- Aplicado a este pipeline: esto se puede implementar con lo que ya es dependencia (scikit-learn) vía una matriz de co-asociación (cuántas veces dos columnas caen en el mismo cluster a través de varias particiones/vistas) seguida de `AgglomerativeClustering` sobre esa matriz de consenso — sin necesidad de elegir un α/β/γ/δ combinado a priori. Es la propuesta más cara de implementar de este documento (ver ranking final).

### 4.3 Ajuste fino contrastivo del encoder: documentado, pero en tensión directa con la restricción de "no supervisado en inferencia"

- La documentación oficial de sentence-transformers recomienda `MultipleNegativesRankingLoss` como *"elección de default recomendada habitualmente"* para pares (ancla, positivo) **sin etiquetas de clase explícitas** — es decir, no requiere una clasificación completa, solo pares de qué-va-con-qué. Otras opciones (`ContrastiveLoss`, `CosineSimilarityLoss`) sí requieren etiquetas binarias o continuas de similitud por par. Fuente: [sbert.net — Loss Overview](https://sbert.net/docs/sentence_transformer/loss_overview.html)
- SetFit demuestra que un ajuste fino contrastivo con un conjunto **pequeño** de pares etiquetados (decenas a cientos de ejemplos, "few-shot"), seguido de una cabeza de clasificación ligera, alcanza resultados comparables a otras técnicas few-shot con órdenes de magnitud menos parámetros/tiempo de entrenamiento. Fuente: [Efficient Few-Shot Learning Without Prompts / SetFit (arXiv:2209.11055)](https://arxiv.org/abs/2209.11055)
- **Por qué esto queda fuera de alcance bajo las reglas de este proyecto, no por falta de evidencia:** ajustar finamente el encoder usando (aunque sea un subconjunto de) las 243 etiquetas manuales, incluso si el clustering en sí sigue corriendo sin ver etiquetas por columna, introduce supervisión dentro del artefacto (el encoder) que se usa para producir el resultado no supervisado reportado — es conceptualmente el mismo problema que separa `health_report.py` (supervisado) de `experiments.py` (no supervisado) en este repo. Se documenta aquí porque la pregunta de investigación lo pedía explícitamente, pero **no se recomienda como candidato a probar** sin antes decidir, como equipo, si el resultado reportado seguiría llamándose "no supervisado".

---

## Tema 5: Señal a nivel de tabla vía features agregadas, sin reglas

### 5.1 Precedente en la literatura de ML: funciones permutation-invariant sobre conjuntos ("Deep Sets")

- El paper fundacional de esta línea caracteriza formalmente *"las funciones invariantes a permutación"* sobre conjuntos y muestra que *"esta caracterización teórica revela una estructura especial que permite diseñar una arquitectura de red profunda que puede operar sobre conjuntos"* — agregando representaciones a nivel de instancia (p. ej. por suma o promedio) en una representación a nivel de grupo, independiente del orden de los elementos. El paper aplica esto a estimación de estadísticas poblacionales, clasificación de nubes de puntos, expansión de conjuntos y **detección de outliers**. Fuente: [Deep Sets (arXiv:1703.06114)](https://arxiv.org/abs/1703.06114)
- Traducido a este pipeline: cada columna ya es una instancia con su propio vector φ(aⱼ); la tabla a la que pertenece es el "conjunto" del que forma parte. Deep Sets da el precedente de que agregar estadísticas del conjunto (aquí: de las columnas de la misma tabla) y anexarlas al vector de cada instancia es una técnica de ML establecida, no una regla ad-hoc — mientras la agregación sea una función estadística/numérica (conteo, proporción, varianza) y no un identificador de tabla en sí.

### 5.2 Qué agregar concretamente, y el riesgo documentado de convertirlo en fuga de identidad

- Candidatos puramente numéricos, calculables sin ver las etiquetas: número de columnas de la tabla, proporción de columnas nullable, diversidad de tipos de dato (p. ej. conteo de tipos distintos / número de columnas), tasa de nombres genéricos (columnas que matchean un patrón como `COL_\d+` sin usar la etiqueta que ese patrón intenta predecir), proporción de columnas con comentario. Todos son estadísticas de la propia tabla, no un one-hot del nombre de tabla.
- El propio `AGENTS.md` de este repo ya documenta el riesgo relevante: sin el comentario de tabla, α=1.00 cae de ARI 0.3748 a 0.2173 en el corpus completo — buena parte de la ganancia actual del documento **ya** viene de que el comentario de tabla filtra identidad de tabla, no de que el modelo generalice el concepto de anti-patrón. Añadir un vector agregado por tabla calculado ingenuamente correría el mismo riesgo a otra escala (con 23 tablas y 153-243 columnas, un vector agregado casi único por tabla es funcionalmente un identificador de tabla). La validación honesta tiene que seguir siendo `--without-giants` más, idealmente, una verificación de que el vector agregado no colapsa en 23 valores triviales (el mismo diagnóstico de `representation_degeneracy` que el repo ya usa para el bloque estructural).

---

## Tema 6: Trabajo previo en detección de anti-patrones de esquema con embeddings o ML

- Búsqueda dirigida en arXiv/ACM/GitHub: **no se encontró ningún paper que combine (a) extracción de metadatos de esquema relacional/Oracle, (b) embeddings de oraciones o representaciones aprendidas, y (c) clustering o clasificación no supervisada de anti-patrones de esquema.** Se dice esto explícitamente en vez de forzar una cita marginal, tal como pide el encargo.
- Lo que sí existe y es adyacente, pero no es lo mismo:
  - **Detección de "code smells"/anti-patrones de código con ML/embeddings** (código fuente, no DDL/metadatos de esquema): usa representaciones como code2vec, code2seq, CodeBERT o CodeT5 sobre texto de programas, con métodos de ensemble y transfer learning entre datasets heterogéneos. Es metodológicamente cercano (embeddings + clasificación/clustering de "mal diseño") pero opera sobre un artefacto distinto (código fuente vs. metadatos de catálogo). Fuentes: [A Machine-learning Based Ensemble Method For Anti-patterns Detection (arXiv:1903.01899)](https://arxiv.org/pdf/1903.01899), búsqueda general sobre detección de code smells con embeddings (CodeBERT/CodeT5), sin arXiv ID único verificado en esta pasada — tratar como pista de búsqueda, no como cita puntual.
  - **Detección de anti-patrones de esquema relacional existente es predominantemente basada en reglas/heurísticas**, no en ML. Ejemplo verificado: un detector de "Database Schema Smell" público en GitHub que *"parsea DDL SQL crudo en un AST, identifica 15+ anti-patrones de diseño relacional... y genera DDL de migración automatizada"* — es un analizador estático basado en reglas, exactamente el enfoque que `rule-engine` (la rama hermana de este repo) ya representa, no una técnica de ML nueva. Fuente: repositorio [database-schema-smell-dectector (GitHub)](https://github.com/GouthamKrishna7/database-schema-smell-dectector), citado como evidencia de que el estado del arte de herramientas en este nicho sigue siendo basado en reglas, no como paper de investigación.
  - El patrón EAV específicamente tiene una literatura extensa mayormente **cualitativa/de diseño** (guías de arquitectura de bases de datos, no ML) sobre por qué es un anti-patrón y cuándo es un compromiso aceptable — no se encontró ningún trabajo de detección automática de EAV vía ML/embeddings.
  - Se localizó un paper con título directamente relevante, *"Measuring and understanding database schema quality"* (ACM), pero el acceso devolvió HTTP 403 (paywall) en esta sesión — no se pudo verificar su contenido ni su metodología (ML vs. métricas manuales), así que no se cita ninguna afirmación sobre él. Queda como pista concreta para una sesión con acceso institucional a ACM Digital Library, no como fuente usada aquí.

---

## Candidatos a probar, en orden de costo/beneficio

1. **(Costo mínimo) Barrer `cluster_selection_method` (`eom` vs `leaf`) y `min_samples` (2, 3, 5) en HDBSCAN.** Ambos parámetros ya existen en `ClusterEngine._fit_hdbscan`; solo falta exponerlos en el barrido de `evaluation/experiments.py`. Justificación (Tema 2): la documentación oficial describe `eom` como sesgado a pocos clusters grandes y `leaf` a muchos clusters finos y homogéneos — el corpus objetivo tiene 10 etiquetas y hoy produce k≈16-19 solo con `eom`, nunca se ha probado `leaf`. Además, `min_samples=2 < min_cluster_size=5` es hoy una elección no documentada como default, y la propia guía de HDBSCAN indica que el default sería 5 (igual a `min_cluster_size`) — vale la pena confirmar que 2 sigue siendo mejor con un barrido explícito en vez de asumirlo.

2. **(Costo bajo) Sustituir el modelo de embeddings por `intfloat/multilingual-e5-small`, y por separado por `paraphrase-multilingual-mpnet-base-v2`/`intfloat/multilingual-e5-base`.** El primero es un cambio "gratis" de tamaño (mismo backbone de 384-dim, 12 capas, que el modelo actual, según su model card) pero con preentrenamiento contrastivo específico para similitud/clustering en vez de solo paráfrasis; los otros dos son un salto a 768-dim con mejores scores de clustering MTEB reportados en su propia tarjeta, a mayor costo de cómputo. Cero dependencias nuevas: `BERTEmbedder` ya usa `AutoModel`/`AutoTokenizer` genérico (Tema 1). Recordar probar el prefijo `"query: "` de la familia e5 (Tema 1.3), porque su ausencia podría penalizar injustamente a esos modelos en la comparación.

3. **(Costo medio) Probar `reducer="umap"` en vez de `"pca"` en el `_cluster` hardcodeado de `experiments.py`, con `min_dist=0.0` y `n_neighbors` bajo (5-15, no el 30 recomendado para datasets grandes).** `umap-learn` ya es dependencia instalada y `DimensionalityReducer` ya soporta el método; el cambio es de dos líneas. Pero la propia documentación de UMAP advierte explícitamente que la combinación UMAP+HDBSCAN es *"algo controvertida"* porque UMAP no preserva bien la densidad y puede crear "falsos desgarros" en clusters (Tema 3) — tratar el resultado con escepticismo cualitativo (inspección de qué columnas terminan juntas) y no solo como una cifra de ARI/AMI que sube o baja.

4. **(Costo medio-alto) Añadir un bloque de features agregadas por tabla** (conteo de columnas, proporción nullable, diversidad de tipos, tasa de nombres genéricos, proporción con comentario) como quinto bloque de `FeatureBuilder`, normalizado igual que los demás con `normalize="block"`. Justificación (Tema 5): es el único candidato de esta lista dirigido específicamente a `giant_table`/`eav`/`polymorphic`, que `AGENTS.md` ya documenta como propiedades de tabla que ningún embedding por-columna puede ver por sí solo. Riesgo documentado y ya medido en este mismo repo (el salto de ARI al quitar el comentario de tabla): validar obligatoriamente con `--without-giants` y con un chequeo de degeneración tipo `representation_degeneracy` para confirmar que el bloque agregado no colapsa en ~23 valores casi únicos (fuga de identidad de tabla con otro nombre).

5. **(Costo alto, más especulativo) Fusión tardía (consensus/late fusion) entre un clustering solo-documento y uno solo-estructura, en vez de concatenación ponderada con α/β/γ/δ.** Implementable con lo que ya es dependencia (matriz de co-asociación + `AgglomerativeClustering` de scikit-learn), justificado por la literatura de multi-view clustering (Tema 4) como alternativa establecida a la fusión temprana por concatenación. Es la propuesta que más código nuevo requiere y la que menos precedente directo tiene dentro de este pipeline — dejarla para después de agotar 1-4.

*Fuera de la lista, deliberadamente:* el ajuste fino contrastivo del encoder con las etiquetas manuales (Tema 4.3) está documentado y sería barato de implementar (SetFit, `MultipleNegativesRankingLoss`), pero se excluye de este ranking porque tensiona directamente la restricción de "no supervisado en inferencia" que este documento debe respetar — cualquier decisión de perseguirlo debería ser explícita, no colarse como una variante más de un barrido.

### Resultado medido del candidato #1 (2026-09-18)

Implementado: `cluster_selection_method` se agregó a `ClusterEngine._fit_hdbscan` (no existía
antes pese a que este documento asumía que sí) y se enhebró como parámetro nombrado por
`run_clustering`/`_cluster`/`_score_phi` en `evaluation/experiments.py`. Barrido sobre la
representación documento (α=1.00), evaluación honesta (sin las dos tablas gigantes, 153
columnas):

| cluster_selection_method | min_samples |    ARI |    NMI |    AMI | Accuracy | F1-macro |  k |
| ------------------------- | -----------: | -----: | -----: | -----: | -------: | -------: | -: |
| eom                        |            2 | 0.0938 | 0.3754 | 0.2464 |   0.6732 |   0.3676 | 17 |
| eom                        |            3 | 0.1107 | 0.3781 | 0.2497 |   0.6797 |   0.4003 | 17 |
| eom                        |            5 | 0.0478 | 0.2883 | 0.1590 |   0.6013 |   0.1629 | 14 |
| leaf                       |            2 | 0.0938 | 0.3754 | 0.2464 |   0.6732 |   0.3676 | 17 |
| leaf                       |            3 | 0.1107 | 0.3781 | 0.2497 |   0.6797 |   0.4003 | 17 |
| leaf                       |            5 | 0.0478 | 0.2883 | 0.1590 |   0.6013 |   0.1629 | 14 |

Hallazgos:

- **`cluster_selection_method` no cambió nada** — `eom` y `leaf` dan resultados idénticos fila
  por fila en este corpus. Negativo pero honesto: a esta escala (153-243 puntos) el árbol
  condensado de HDBSCAN aparentemente selecciona los mismos clusters por ambos criterios. No se
  investigó más a fondo por qué (fuera de alcance de "probar el candidato más barato").
- **`min_samples=3` sí gana**, en las cinco métricas, sobre 2 y sobre 5. Adoptado como nuevo
  default (`HDBSCAN_MIN_SAMPLES = 3` en `evaluation/experiments.py`, antes 2).
- Efecto en el corpus completo (243 columnas, con las gigantes): mixto por representación —
  documento y anclas mejoran, nombre-solo empeora. Se priorizó la evaluación sin gigantes, que
  AGENTS.md ya trata como la que vale, sobre el corpus completo (parcialmente inflado por fuga de
  identidad de tabla, ver AGENTS.md).
- Costo real: al cambiar un hiperparámetro de clustering compartido por todas las
  representaciones, se pierde la propiedad (documentada antes en AGENTS.md) de que nombre-solo y
  anclas reproducían la tabla histórica cifra por cifra — ese chequeo de cordura dependía de que
  ningún hiperparámetro de clustering cambiara respecto al run histórico. Se documenta el
  trade-off en AGENTS.md en vez de ocultarlo.

### Resultado medido del candidato #2 (2026-09-18)

Implementado: `scripts/build_embeddings.py::build()` gana un parámetro `text_prefix` (CLI:
`--query-prefix`), prependido a cada documento antes de codificar — necesario para reproducir el
uso documentado de la familia `intfloat/multilingual-e5-*` (Tema 1.3). `--model` ya existía como
flag, así que cambiar de encoder no necesitó ningún otro cambio de código.

Se generaron pickles con `intfloat/multilingual-e5-small` (con y sin el prefijo `"query: "`,
para aislar su efecto), `paraphrase-multilingual-mpnet-base-v2` y `intfloat/multilingual-e5-base`,
y se compararon contra el `paraphrase-multilingual-MiniLM-L12-v2` actual con
`cli experiment --without-giants --ablation` (la evaluación honesta, 153 columnas):

| modelo                              | dim | ARI    | NMI    | AMI    | Accuracy | F1-macro |  k |
| ------------------------------------ | --: | -----: | -----: | -----: | -------: | -------: | -: |
| MiniLM-L12-v2 (actual)                | 384 | 0.1107 | 0.3781 | 0.2497 |   0.6797 |   0.4003 | 17 |
| multilingual-e5-small (con prefijo)   | 384 | 0.0680 | 0.3564 | 0.2177 |   0.6536 |   0.3671 | 18 |
| multilingual-e5-small (sin prefijo)   | 384 | 0.0690 | 0.3586 | 0.2204 |   0.6601 |   0.3688 | 18 |
| mpnet-base-v2                         | 768 | 0.0713 | 0.3401 | 0.2036 |   0.6405 |   0.3892 | 17 |
| multilingual-e5-base (con prefijo)    | 768 | 0.0538 | 0.3398 | 0.1974 |   0.6405 |   0.3609 | 18 |

Hallazgos:

- **El modelo actual le gana a las tres alternativas en las cinco métricas**, en la evaluación
  sin gigantes y también en el corpus completo (mismo orden, no se muestra la tabla por brevedad).
  Resultado negativo honesto: ni el modelo del mismo tamaño con preentrenamiento contrastivo
  (e5-small) ni los modelos más grandes con mejor score de clustering en su propio MTEB card
  (mpnet, e5-base) mejoran este corpus específico.
- **El prefijo `"query: "` no explica la diferencia** — con y sin prefijo, e5-small da
  prácticamente el mismo resultado (ARI 0.0680 vs 0.0690, dentro del ruido). No es un artefacto
  de implementación; el modelo genuinamente rinde peor aquí.
- Lectura plausible, no verificada a fondo: el corpus son oraciones cortas y muy estructuradas en
  español ("tabla X, columna Y, tipo Z..."), más cercanas al objetivo de paráfrasis con el que se
  entrenó MiniLM-L12-v2 que al objetivo de recuperación contrastiva de la familia e5 o al espacio
  semántico general de mpnet — pero esto queda como hipótesis, no como hallazgo confirmado contra
  fuente primaria.
- **Sin cambios al default**: `model_name` en `BERTEmbedder`/`build_embeddings.py` se queda en
  `paraphrase-multilingual-MiniLM-L12-v2`. El flag `--query-prefix` se conserva en el código (no
  tiene costo — default `""`, no-op para el modelo actual) porque deja la infraestructura lista
  para probar otra familia de modelos en el futuro sin tener que reimplementarlo.
- Los 5 pickles de prueba (`intermediate_docs_e5small*.pkl`, `*_mpnet.pkl`, `*_e5base.pkl`) se
  borraron tras la comparación — están gitignored y el hallazgo (negativo) ya quedó documentado
  aquí, no hacía falta conservarlos.

### Resultado medido del candidato #3 (2026-09-18)

Implementado: `evaluation/experiments.py::run_clustering`/`_cluster`/`_score_phi` ganan un
parámetro `reducer_kwargs: dict | None`, enhebrado a `DimensionalityReducer(...,
**reducer_kwargs)`. Sin esto, `reducer="umap"` ya era seleccionable pero corría siempre con los
defaults del reductor (`n_neighbors=15, min_dist=0.1`, pensados para datasets mucho más grandes
— Tema 3.2), sin forma de pasarle `min_dist=0.0` ni un `n_neighbors` bajo desde el harness de
experimentos.

Barrido sobre la representación documento (α=1.00), `min_dist=0.0` fijo, `n_neighbors` ∈
{5, 10, 15} × salida de UMAP a 20 o 2 dimensiones, evaluación honesta (sin gigantes):

| reducer                    |    ARI |    NMI |    AMI | Accuracy | F1-macro |  k |
| --------------------------- | -----: | -----: | -----: | -------: | -------: | -: |
| **pca (actual, n=20)**       | 0.1107 | 0.3781 | 0.2497 |   0.6797 |   0.4003 | 17 |
| umap nn=5, d=20              | 0.0569 | 0.3579 | 0.2142 |   0.6536 |   0.3835 | 19 |
| umap nn=5, d=2                | 0.0505 | 0.3452 | 0.1987 |   0.6340 |   0.3745 | 19 |
| umap nn=10, d=20              | 0.0508 | 0.3457 | 0.1994 |   0.6340 |   0.3745 | 19 |
| umap nn=10, d=2               | 0.0454 | 0.3306 | 0.1862 |   0.6144 |   0.3675 | 18 |
| umap nn=15, d=20              | 0.0533 | 0.3485 | 0.2025 |   0.6471 |   0.3808 | 19 |
| umap nn=15, d=2 (mejor UMAP)   | 0.0685 | 0.3766 | 0.2368 |   0.6732 |   0.3919 | 19 |

En el corpus completo la brecha es aún mayor: PCA ARI 0.3861 vs el mejor UMAP (nn=15, d=2)
ARI 0.1553.

Hallazgos:

- **PCA le gana a las siete configuraciones de UMAP probadas**, en ambas evaluaciones. Ninguna
  combinación de `n_neighbors`/dimensión de salida se acerca al baseline actual.
- Esto confirma en la práctica el caveat que la propia documentación de UMAP ya daba por
  adelantado (Tema 3): la combinación UMAP+HDBSCAN es "algo controvertida" porque UMAP no
  preserva bien la densidad, justo la propiedad de la que depende un algoritmo basado en
  densidad como HDBSCAN. Sobre 153-243 puntos (muy por debajo de la escala en la que la propia
  guía de UMAP da sus ejemplos), el efecto es claramente negativo, no neutro.
- **Sin cambios al default**: `reducer="pca"` se queda. `reducer_kwargs` se conserva en el código
  (sin costo, no-op si no se pasa) como infraestructura para futuros barridos de UMAP u otro
  reductor no lineal, sin tener que volver a tocar `experiments.py`.

### Resultado medido del candidato #4 (2026-09-18)

Implementado: nuevo módulo `features/table_aggregates.py` (`build_table_block(schema,
column_index) -> (N,5)`: log1p(nº columnas), % nullable, diversidad de tipos, % nombres
genéricos, % con comentario de columna — todo calculado desde la estructura, sin nombre de tabla
ni reglas escritas a mano). `FeatureBuilder` gana un quinto bloque opcional `e_table` con peso
`epsilon` (default 0.0, mismo patrón que `e_stat`/`delta`: presente pero con contribución cero si
no se pide). `Dataset`/`load_dataset` calculan `e_table` automáticamente desde `schema` — que ya
vive en todos los pickles existentes — así que ningún pickle necesitó regenerarse.

**Chequeo de degeneración obligatorio (Tema 5.2) antes de confiar en cualquier número:** de las
23 tablas del corpus reconstruido, el vector agregado de 5 dimensiones es **distinto en 21** —
solo 2 colisiones exactas (`AUDITORIA_LOG`≡`EMPLEADOS_HISTORIAL`, `CATEGORIAS`≡`METADATA`). Esto
es casi tan identificador como el nombre de tabla mismo — el riesgo que este mismo documento
advertía por adelantado. Dato a favor: `BACKUP_DATOS` y `TABLA_BASE_DATOS` (las dos gigantes)
quedan con vectores casi idénticos entre sí por razones estructurales genuinas (muchas columnas,
casi todas nullable, tipos poco diversos, nombres genéricos, cero comentarios) — exactamente la
señal que este candidato buscaba, no una coincidencia de identidad.

Barrido de `epsilon` sobre la representación documento (α=1.00), evaluación honesta (sin
gigantes):

| epsilon |    ARI |    NMI |    AMI | Accuracy | F1-macro |  k |
| ------- | -----: | -----: | -----: | -------: | -------: | -: |
| 0.00 (baseline) | 0.1107 | 0.3781 | 0.2497 |   0.6797 |   0.4003 | 17 |
| 0.10    | 0.1028 | 0.3681 | 0.2375 |   0.6667 |   0.3639 | 17 |
| 0.15    | 0.1134 | 0.3886 | 0.2624 |   0.6797 |   0.3782 | 17 |
| 0.20    | 0.1054 | 0.3807 | 0.2527 |   0.6797 |   0.3934 | 17 |
| **0.25**| **0.1158** | **0.3925** | **0.2670** | **0.6863** | **0.4040** | 17 |
| 0.30    | 0.1158 | 0.3925 | 0.2670 |   0.6863 |   0.4040 | 17 |
| 0.35    | 0.1042 | 0.3728 | 0.2429 |   0.6732 |   0.3760 | 17 |
| 0.40    | 0.1032 | 0.3673 | 0.2360 |   0.6667 |   0.3617 | 17 |

Hallazgos:

- **Mejora real pero modesta**: `epsilon≈0.25-0.30` gana en las cinco métricas frente al
  baseline sin bloque de tabla (ARI +0.0051, ~4.6% relativo — mucho más chico que la mejora del
  candidato #1). Es un pico genuino, no ruido: sube de forma monótona hasta 0.25-0.30 y vuelve a
  bajar después, con una meseta plana exactamente en el óptimo (0.25 y 0.30 dan el mismo
  resultado).
- **La mejora no puede venir de re-identificar las dos tablas gigantes** — están excluidas de
  esta evaluación (`--without-giants`) — así que lo que sea que epsilon aporta viene de las otras
  21 tablas, no de las dos cuyo riesgo de fuga ya estaba medido. Eso no descarta que parte de la
  mejora sea el mismo mecanismo de fuga aplicado a otro par de tablas (p. ej. `AUDITORIA_LOG`/
  `EMPLEADOS_HISTORIAL`, que colisionan exactamente) — no se investigó eso a más profundidad.
- **Decisión: `epsilon` se queda en 0.0 por defecto, no se adopta como nuevo default.** Mismo
  criterio que `--mismatch` (AGENTS.md): una mejora real pero pequeña, sobre un bloque con un
  riesgo de fuga de identidad ya medido y no completamente descartado, no debe volverse
  comportamiento por defecto silencioso. Queda disponible vía la librería
  (`evaluate(nombre, dataset, {"alpha":1.0,...,"epsilon":0.25})`) para quien quiera seguir
  investigando esta dirección — por ejemplo, separando qué columnas específicas cambian de
  cluster al subir epsilon, para confirmar si son realmente columnas `giant_table`/`eav`/
  `polymorphic` las que se benefician, o si el efecto es más difuso.

---

## Referencias

**Sentence-transformers / Hugging Face / MTEB:**
- MTEB: Massive Text Embedding Benchmark (arXiv:2210.07316): https://arxiv.org/abs/2210.07316
- MMTEB: Massive Multilingual Text Embedding Benchmark (arXiv:2502.13595): https://arxiv.org/abs/2502.13595
- sbert.net — Pretrained Models: https://sbert.net/docs/sentence_transformer/pretrained_models.html
- sbert.net — Loss Overview: https://sbert.net/docs/sentence_transformer/loss_overview.html
- model card paraphrase-multilingual-MiniLM-L12-v2: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- model card paraphrase-multilingual-mpnet-base-v2: https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2
- model card intfloat/multilingual-e5-small: https://huggingface.co/intfloat/multilingual-e5-small
- model card intfloat/multilingual-e5-base: https://huggingface.co/intfloat/multilingual-e5-base
- model card BAAI/bge-m3: https://huggingface.co/BAAI/bge-m3
- SetFit — Efficient Few-Shot Learning Without Prompts (arXiv:2209.11055): https://arxiv.org/abs/2209.11055

**HDBSCAN / scikit-learn:**
- hdbscan docs — Parameter Selection for HDBSCAN*: https://hdbscan.readthedocs.io/en/latest/parameter_selection.html
- scikit-learn — sklearn.cluster.HDBSCAN: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html

**UMAP:**
- umap-learn docs — Using UMAP for Clustering: https://umap-learn.readthedocs.io/en/latest/clustering.html
- umap-learn docs — FAQ: https://umap-learn.readthedocs.io/en/latest/faq.html
- UMAP: Uniform Manifold Approximation and Projection (arXiv:1802.03426): https://arxiv.org/abs/1802.03426

**Fusión multi-vista / representación de conjuntos:**
- A Survey on Multi-View Clustering (arXiv:1712.06246): https://arxiv.org/abs/1712.06246
- Late Fusion Multi-view Clustering via Global and Local Alignment Maximization (arXiv:2208.01198): https://arxiv.org/abs/2208.01198
- Deep Sets (arXiv:1703.06114): https://arxiv.org/abs/1703.06114

**Detección de anti-patrones de esquema / code smells (Tema 6):**
- A Machine-learning Based Ensemble Method For Anti-patterns Detection (arXiv:1903.01899): https://arxiv.org/pdf/1903.01899
- database-schema-smell-dectector (GitHub, herramienta basada en reglas, no paper): https://github.com/GouthamKrishna7/database-schema-smell-dectector
- "Measuring and understanding database schema quality" (ACM DL, acceso bloqueado por paywall en esta sesión, no verificado): https://dl.acm.org/doi/10.1145/3183519.3183529
