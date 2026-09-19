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

### Resultado medido del candidato #5 (2026-09-18)

Implementado: nuevo módulo `clustering/late_fusion.py` (`co_association_matrix`,
`consensus_clustering`) y `evaluation/experiments.py::evaluate_late_fusion` — clustera cada
vista por separado (misma reducción/HDBSCAN que el resto del pipeline), arma la matriz de
co-asociación (fracción de vistas en las que cada par de columnas cae en el mismo cluster; el
ruido de HDBSCAN nunca cuenta como acuerdo, ni siquiera entre dos puntos de ruido) y corta un
árbol de `AgglomerativeClustering(metric="precomputed")` a un `distance_threshold` — sin elegir
ningún α/β/γ/δ combinado de antemano, y sin usar la cuenta real de etiquetas para fijar
`n_clusters` (eso rompería el carácter no supervisado).

Barrido de `distance_threshold` sobre dos vistas (documento α=1.00, estructura `STRUCTURE_ONLY`)
y luego tres (+ tabla-agregada del candidato #4, ε=1.00), evaluación honesta (sin gigantes):

| vistas | threshold |     ARI |    NMI |    AMI | F1-macro |  k |
| ------ | --------: | ------: | -----: | -----: | -------: | -: |
| **documento solo (baseline, fusión temprana)** | — | **0.1107** | **0.3781** | **0.2497** | **0.4003** | 17 |
| documento + estructura | 0.1 – 0.5 | 0.0134 | 0.4107 | 0.0789 | 0.7877 | 88 |
| documento + estructura | 0.75      | -0.0218 | 0.2664 | 0.0993 | 0.3342 | 22 |
| documento + estructura | 0.9       | -0.0376 | 0.2405 | 0.1073 | 0.3032 | 16 |
| documento + estructura + tabla | 0.15 – 0.3 | 0.0127 | 0.4104 | 0.0757 | 0.7877 | 89 |
| documento + estructura + tabla | 0.4 – 0.6  | 0.0460 | ~0.389 | ~0.189 | 0.5164 | ~38 |
| documento + estructura + tabla | 0.7        | 0.0752 | 0.3846 | 0.2571 | 0.3781 | 17 |
| documento + estructura + tabla | 0.85       | -0.0285 | 0.2692 | 0.1565 | 0.1881 | 11 |

Hallazgos:

- **Ninguna configuración le gana al baseline de fusión temprana** — ni con 2 vistas ni con 3,
  en ningún punto del barrido de threshold. El mejor resultado (3 vistas, threshold=0.7) da ARI
  0.0752, todavía por debajo de 0.1107.
- **El umbral estricto fragmenta, no afina**: con threshold bajo (requiere acuerdo en *todas*
  las vistas para fusionar), el resultado es una partición mucho más fina que cualquier vista
  individual — k=88-89 sobre 153 columnas, muchos clusters diminutos. El F1-macro se ve
  artificialmente alto (0.7877) precisamente por eso: el nombrado por voto mayoritario acierta
  trivialmente en clusters de 1-2 columnas. El ARI (la métrica honesta sobre la partición) lo
  delata: 0.0127-0.0134, muy por debajo del baseline. Exactamente el caso que AGENTS.md ya
  advierte de F1 como cota superior engañosa.
- **El umbral laxo (≥0.75-0.9, fusiona si *alguna* vista coincide) tampoco ayuda** — funde
  columnas que solo una vista débil (estructura, que ya colapsa mucho — ver
  `representation_degeneracy`) puso juntas, y el ARI cae por debajo de cero.
- Con solo 2 vistas, la matriz de co-asociación solo toma 3 valores (0, 0.5, 1), lo que hace al
  barrido de threshold poco expresivo entre 0.1 y 0.5 (idéntico resultado en todo ese rango,
  visible en la tabla). Agregar una tercera vista (tabla-agregada) da más resolución (0, 0.33,
  0.67, 1) y un resultado algo mejor en el punto óptimo (0.0752 vs 0.0134), pero sigue sin
  acercarse al baseline.
- **Sin cambios al pipeline por defecto.** `evaluate_late_fusion`/`consensus_clustering` quedan
  en el código, testeados, como una alternativa disponible — útil si en el futuro se agregan más
  vistas genuinamente informativas (más de 3), donde una matriz de co-asociación más fina podría
  comportarse distinto. Con las vistas disponibles hoy, la fusión temprana (concatenación
  ponderada) sigue siendo la mejor opción medida.

Con esto se cierran los 5 candidatos de este documento: 1 adoptado como nuevo default
(`min_samples=3`), 1 disponible opt-in con reserva documentada (`epsilon`/tabla-agregada), 3
resultados negativos honestos (modelo de embeddings, UMAP, fusión tardía).


### Candidato #6 (2026-09-19): representación de conflicto — el problema era qué representa el vector

Los cinco candidatos anteriores cambiaron el encoder, el reductor, los hiperparámetros o la
fusión. Ninguno cambió **qué representa el vector**. Este candidato parte de un diagnóstico, no
de una lista de técnicas.

#### Diagnóstico: los clusters del documento son las tablas

Sin las gigantes (153 columnas), documento α=1.00, `mcs=5 ms=3`:

| se compara la partición contra… |   ARI |   AMI |
| -------------------------------- | ----: | ----: |
| la **tabla** de cada columna     | 0.593 | 0.754 |
| la **etiqueta** de anti-patrón   | 0.111 | 0.250 |

14 de 16 clusters son una sola tabla. Las etiquetas que "acierta" son las que coinciden con una
tabla entera: `TBL_DATOS`→`inconsistent_naming`, `CONFIGURACION`→`eav`, `REPORTES`→
`reserved_words`. En el embedding crudo, el 84 % de los 5 vecinos más cercanos de una columna son
de su misma tabla. Para `wrong_data_types`, que está repartida en 15 tablas, solo el 20 % comparte
su etiqueta.

Quitar la identidad de tabla no deja nada debajo:

| variante del documento (sin gigantes)          |    ARI |
| ---------------------------------------------- | -----: |
| documento completo                             |  0.111 |
| sin comentario de tabla                        | -0.008 |
| centrado por tabla (restar la media por tabla) |  0.041 |
| oración solo de columna (sin tabla)            |  0.008 |
| quitar las 1–5 primeras componentes principales | 0.070–0.079 |

La causa es de representación. Una oración **promedia** sus facetas. 186 de 243 columnas comparten
la faceta "texto de longitud variable de hasta 255 caracteres, admite nulos", así que la frase del
tipo domina el vector. Pero un anti-patrón como `wrong_data_types` es un **conflicto entre
facetas**: el nombre dice fecha y el tipo dice texto. El mean-pooling diluye justo eso. El techo
supervisado lo confirma: un RandomForest con 5 folds sobre el embedding del documento llega a
F1-macro 0.49. Con solo estructura llega a 0.13. El límite estaba en la representación, no en el
algoritmo de clustering.

#### El cambio: BERT como instrumento de medida, no como 384 coordenadas

`ConflictBlock` (en `features/semantic_anchors.py`) le hace al encoder dos preguntas por separado
y resta las respuestas:

1. **Qué espera el nombre.** Se usa el coseno del nombre desnudo de la columna contra 6 anclas de
   familia de almacenamiento: fecha, importe, referencia, texto, booleano y binario. Un softmax a
   temperatura 0.05 lo convierte en distribución. Es zero-shot, sin etiquetas.
2. **Qué declara el tipo.** Es la misma distribución de 6 familias, leída del tipo Oracle. Una FK
   añade `referencia` y un `CHAR(1)` añade `booleano`.

El bloque tiene 13 dimensiones: la diferencia (6), la familia declarada (6) y la entropía de la
expectativa (1). Una columna cuyo nombre y tipo concuerdan cae cerca del origen, sea cual sea su
tema. Cada tipo de conflicto se desplaza en su propia dirección. Esa es la geometría que un
algoritmo de densidad puede aprovechar.

`FeatureBuilder` lo pesa con `zeta`, que vale 0.0 por defecto. Con ese valor todo experimento
anterior se reproduce cifra por cifra, verificado corriendo la ablación en el commit anterior y
en este. Los pesos nuevos son `CONFLICT` (conflicto 1.0 y restricciones 0.5, sin embeddings) y
`CONFLICT_FUSED` (añade 0.2 de documento y 0.3 de tabla agregada).

#### Protocolo nuevo: el punto de operación se elige a ciegas

Cada decisión de hiperparámetros anterior de este repo se tomó mirando el ARI contra las
etiquetas, incluido `min_samples=3` del candidato #1. Eso es ajustar sobre el conjunto de prueba.
`stability_sweep` recorre una rejilla de HDBSCAN con `min_cluster_size` ∈ {5, 6, 7, 8, 10} y
`min_samples` ∈ {2, 3, 5}. Elige la celda con mayor `relative_validity_` de HDBSCAN, una
aproximación de DBCV que nunca ve las etiquetas. La rejilla completa se publica. La columna
**ARI~tbl** mide el ARI de la partición contra la tabla de cada columna: cuánto de un clustering
es reconocimiento de tabla.

#### Resultados, sin gigantes (153 columnas), ambas representaciones elegidas a ciegas

| representación              | celda elegida |    ARI |    AMI | F1 (cota) |  k | ARI~tbl |
| --------------------------- | ------------- | -----: | -----: | --------: | -: | ------: |
| documento α=1.00            | mcs=5 ms=5    | 0.0478 | 0.1590 |    0.1629 | 14 |   0.381 |
| **conflicto**               | mcs=6 ms=3    | 0.1276 | 0.2181 |    0.2323 | 11 |   0.010 |
| conflicto fusionado         | mcs=5 ms=2    | 0.1200 | 0.2230 |    0.2324 | 11 |   0.025 |

Sobre la rejilla completa de 15 celdas:

| representación      | ARI mediana | ARI mín–máx   | AMI mediana | ARI~tbl mediana |
| ------------------- | ----------: | ------------- | ----------: | --------------: |
| documento α=1.00    |      0.0647 | 0.036 – 0.111 |      0.1825 |           0.267 |
| conflicto           |      0.1186 | 0.044 – 0.198 |      0.2104 |           0.011 |
| conflicto fusionado |      0.1110 | -0.099 – 0.171 |     0.2101 |           0.017 |

La **mediana** de la rejilla del conflicto (0.1186) supera al **máximo** de la del documento
(0.111). Ese máximo es exactamente la cifra que el repo reportaba como resultado. El conflicto
gana en ARI y AMI y elimina la fuga de tabla, de 0.38 a 0.01.

Rejilla completa del conflicto, sin gigantes (salida de `cli experiment --without-giants --stability`):

| mcs | ms |    ARI |    AMI | F1-macro |  k | ruido | ARI~tbl | validity |
| --: | -: | -----: | -----: | -------: | -: | ----: | ------: | -------: |
|   5 |  2 | 0.1064 | 0.2066 |   0.2308 | 13 |   21% |  0.0258 |   0.2841 |
|   5 |  3 | 0.0865 | 0.2030 |   0.2323 | 13 |   27% |  0.0190 |   0.3914 |
|   5 |  5 | 0.0444 | 0.1833 |   0.1662 |  8 |   48% |  0.0094 |   0.3260 |
|   6 |  2 | 0.1317 | 0.2145 |   0.2308 | 12 |   24% |  0.0204 |   0.3152 |
|   6 |  3 | **0.1276** | **0.2181** | 0.2323 | 11 | 33% | 0.0098 | **0.4631** (elegida) |
|   6 |  5 | 0.0444 | 0.1833 |   0.1662 |  8 |   48% |  0.0094 |   0.3260 |
|   7 |  2 | 0.1317 | 0.2145 |   0.2308 | 12 |   24% |  0.0204 |   0.3152 |
|   7 |  3 | 0.1276 | 0.2181 |   0.2323 | 11 |   33% |  0.0098 |   0.4631 |
|   7 |  5 | 0.0444 | 0.1833 |   0.1662 |  8 |   48% |  0.0094 |   0.3260 |
|   8 |  2 | 0.1659 | 0.2272 |   0.2308 | 11 |   29% |  0.0183 |   0.3861 |
|   8 |  3 | 0.1984 | 0.2409 |   0.2130 |  8 |   22% |  0.0136 |   0.2171 |
|   8 |  5 | 0.0562 | 0.2051 |   0.1662 |  7 |   45% |  0.0165 |   0.3344 |
|  10 |  2 | 0.1302 | 0.2104 |   0.1666 |  5 |   39% |  0.0096 |   0.1643 |
|  10 |  3 | 0.1186 | 0.2112 |   0.1670 |  5 |   39% |  0.0115 |   0.1776 |
|  10 |  5 | 0.1051 | 0.2031 |   0.1662 |  5 |   40% |  0.0109 |   0.1828 |

La mejor celda por etiquetas (`mcs=8 ms=3`, ARI 0.198) **no** es el resultado. Citarla sería
repetir el error de protocolo que este candidato corrige.

Composición de la partición del conflicto con `mcs=5 ms=3`: los clusters cruzan tablas.

- **13 columnas** de 11 tablas, 11 de ellas `wrong_data_types`: FECHA, TIMESTAMP,
  FECHA_NACIMIENTO, FECHA_ALTA… Son fechas guardadas como texto.
- **15 columnas** de 10 tablas, 10 de ellas `wrong_data_types`: SALARIO, PRESUPUESTO, PRECIO,
  STOCK_2… Son importes guardados como texto.
- **8 columnas** de 5 tablas: ACTIVO, FLAG_ACTIVO, FLAG_S_N, FLAG_Y_N… Son los flags, que
  mezclan `self_contradictory` e `inconsistent_naming`.
- **9 columnas** de 6 tablas: REGISTRO_ID, DEPARTAMENTO, PRODUCTO_ID… Son referencias sin FK,
  3 de las 4 `impossible_data`.
- **17 IDs limpios** de 17 tablas, y varios clusters de nombres y descripciones limpios.

#### Corpus completo (243 columnas): el documento sigue ganando, por la razón conocida

| representación      | celda elegida |    ARI |    AMI | ARI~tbl |
| ------------------- | ------------- | -----: | -----: | ------: |
| documento α=1.00    | mcs=5 ms=3    | 0.3861 | 0.4802 |   0.884 |
| conflicto           | mcs=6 ms=3    | 0.1470 | 0.2996 |   0.177 |
| conflicto fusionado | mcs=10 ms=2   | 0.2921 | 0.4168 |   0.381 |

`giant_table` es 90 de 243 columnas y es propiedad de la tabla. El documento la "detecta"
reconociendo las dos tablas: tiene ARI~tbl 0.884. El conflicto no puede verla, porque una columna
de una tabla gigante no tiene conflicto entre nombre y tipo. La versión fusionada recupera parte
a través del bloque de tabla agregada. Esto confirma, no contradice, que `--without-giants` es
la evaluación que vale.

#### Advertencias

- **Sensibilidad a las constantes.** En el prototipo, la temperatura 0.05 superó a 0.03 y a 0.10
  en casi toda la rejilla. Un fraseo corto de las anclas ("fecha u hora", "si o no") colapsó a un
  ARI cercano a 0. Promediar 3 paráfrasis por ancla **no** lo estabilizó. Las constantes de
  `TYPE_FAMILY_ANCHORS` y `EXPECTATION_TEMPERATURE` son parte de la representación. Se eligieron
  mirando el ARI del prototipo, igual que cualquier decisión previa del repo. La elección ciega
  cubre HDBSCAN, no el fraseo.
- **El bloque no se puede z-scorear por dimensión.** Las columnas indicadoras escasas de la
  familia declarada tienen pocos unos. Con z-score por dimensión reciben valores de ±6 y dominan
  toda distancia: medido, HDBSCAN colapsa a k≈5 con ARI 0. `scale_conflict_subblocks` escala por
  sub-bloque con pesos 1.0 : 0.7 : 0.5.
- **La F1 cota superior baja**, de 0.40 a 0.23. El documento la ganaba nombrando clusters que
  eran tablas. ARI y AMI son las métricas honestas.
- **Conflictos que se escapan** en el prototipo. `EMAIL DATE` cae en ruido. `MONTO` sale ambiguo,
  con expectativa 0.49 fecha y 0.38 importe: el encoder no conoce bien "monto".
  `INTENTOS_FALLIDOS` se lee como booleano. `NIVEL` y `PROVEEDORES_ID VARCHAR2` son conflictos que
  el nombre solo no delata.
- **El ground truth pone su propio techo.** `inconsistent_naming` junta tres fenómenos: nombres
  en inglés (`ORDER_ID`), nombres genéricos (`C1`–`C7`) y convenciones de flag mezcladas
  (`FLAG_S_N`, `FLAG_Y_N`, `FLAG_1_0`, `FLAG_T_F`). `clean` incluye `COL_A_BORRAR_1`,
  `DUPLICADO_NOMBRE`, `BACKUP_COLUMN`, `REPETIDA_1` y `STOCK_COPIA`. Ninguna partición puede
  reproducir etiquetas que agrupan cosas distintas. No se tocó ninguna etiqueta.
- **Criterio no alcanzado.** El plan pedía ARI ≥ 0.15 en la celda elegida a ciegas y se quedó en
  0.128. Se reporta tal cual.
- **Informe de salud.** `build_raw_features` añade el bloque crudo. La F1-macro fuera de fold del
  clasificador de referencia pasa de 0.5168 ± 0.090 a 0.5238 ± 0.069, dentro del ruido. No
  empeora, pero tampoco mejora de forma medible.

#### Decisión

`zeta` queda en 0.0 por defecto, así que ningún experimento existente cambia. `--conflict` y
`--stability` quedan como la forma de reportar este resultado. Recomendación: reportar el
conflicto elegido a ciegas como resultado principal sin gigantes, y dejar de citar
`documento, min_samples=3, ARI 0.1107` como resultado sin la nota de que se eligió mirando las
etiquetas. Esa segunda recomendación es una decisión del autor del trabajo, no de este cambio.


### Candidato #7 (2026-09-19): expectativa dura y Ward — quitar la fragilidad del candidato #6

El candidato #6 dejó una cifra ciega de ARI 0.1276 sin gigantes, frágil ante la temperatura y el
fraseo de las anclas. Este candidato midió cinco palancas antes de implementar nada. Implementa
las dos que funcionan y documenta las que no. Las etiquetas no se tocaron, por decisión del autor.

#### Causa raíz de la fragilidad

La expectativa suave es un softmax de los cosenos del nombre contra las 6 familias. Cuando se
aplana, por subir la temperatura o cambiar el fraseo, la familia declarada domina el bloque. El
criterio interno, sea silueta o validez, elige entonces k=2: un corte por tipo declarado con ARI
-0.08. Medido con Ward y k por silueta en 2..30, sin gigantes:

| expectativa suave, fusionado | temp 0.03 | temp 0.05 | temp 0.10 |
| ---------------------------- | --------: | --------: | --------: |
| anclas `orig`                |    0.1735 |    0.1779 | -0.0776 (k=2) |
| anclas `W2`                  |    0.0906 | -0.0776 (k=2) |  0.1084 |
| anclas `W3`                  | -0.0723 (k=3) | -0.0776 (k=2) | 0.0080 |

En el corpus completo, la versión suave colapsa a k=2 incluso con `orig` a 0.05 en cuanto k puede
ir de 2 a 30.

#### Lo que se implementó

1. **Expectativa dura** (`ConflictBlock(mode="hard")`, ahora por defecto). Es un one-hot sobre la
   familia de mayor coseno. La columna de confianza pasa a ser el margen entre el mejor coseno y el
   segundo. No hay temperatura, y ninguno de los tres fraseos colapsa. `mode="soft"` sigue
   disponible para reproducir el candidato #6.
2. **Ward con k por silueta** (`evaluate_blind(..., algorithm="ward")`, k en 2..30). Asigna cada
   columna, así que no hay un bloque de ruido que el scoring cuente como un cluster más. Sobre la
   representación de conflicto, la silueta tiene su máximo global en k=17 en todo el rango 2..40.
   El límite superior no es lo que elige.
3. **HDBSCAN como control**, con la celda elegida por validez relativa y el ruido reasignado a sus
   3 vecinos agrupados (`reassign_noise_knn`).
4. **Robustez sobre fraseos** (`--robustness`). El bloque se construye con las tres redacciones de
   `TYPE_FAMILY_ANCHOR_VARIANTS` y se reporta la media. Las anclas `orig` se eligieron en el
   prototipo del candidato #6 mirando el ARI. Por eso la cifra principal es la media, no `orig`.

#### Resultados, todo elegido a ciegas

Sin gigantes (153 columnas), representación fusionada:

| configuración                      | algoritmo    |    ARI |    AMI |  k | ARI~tbl |
| ---------------------------------- | ------------ | -----: | -----: | -: | ------: |
| documento α=1                      | Ward         | 0.0505 | 0.2105 | 24 |   0.867 |
| documento α=1                      | HDBSCAN+kNN  | 0.0448 | 0.1502 | 13 |   0.655 |
| candidato #6 (suave, `orig`)       | HDBSCAN      | 0.1276 | 0.2181 | 11 |   0.010 |
| **duro, `orig`**                   | Ward         | 0.1693 | 0.2464 | 17 |   0.006 |
| **duro, media de 3 fraseos**       | Ward         | 0.1256 | 0.2004 | ≥17 |  0.002 |
| duro, media de 3 fraseos           | HDBSCAN+kNN  | 0.1288 | 0.1645 | ≥6 |   0.007 |
| suave, media de 3 fraseos, temp 0.05 | Ward       | 0.0076 |      — | 2–17 |    — |

Corpus completo (243 columnas):

| configuración                | algoritmo   |    ARI |    AMI |  k | ARI~tbl |
| ---------------------------- | ----------- | -----: | -----: | -: | ------: |
| documento α=1                | Ward        | 0.3659 | 0.4522 | 23 |   0.963 |
| documento α=1                | HDBSCAN     | 0.3861 | 0.4802 | 19 |   0.884 |
| **duro, `orig`**             | Ward        | 0.4185 | 0.3673 | 15 |   0.300 |
| **duro, media de 3 fraseos** | Ward        | 0.3561 | 0.3729 | ≥15 |  0.380 |

Por fraseo (Ward), sin gigantes: `orig` 0.1693, `W2` 0.0961 y `W3` 0.1115. En el corpus completo:
0.4185, 0.3435 y 0.3062.

#### Lectura honesta

- **Robustez: es la ganancia clara.** Con el mismo protocolo, la media sobre fraseos pasa de 0.0076
  (suave) a 0.1256 (dura). Ningún fraseo cae por debajo de k=17 con Ward.
- **ARI: sube.** Sin gigantes, la media dura (0.1256) duplica al documento elegido a ciegas
  (0.0505). Con `orig` llega a 0.1693. En el corpus completo, `orig` supera al documento (0.4185
  contra 0.3659–0.3861) con un tercio de su fuga de tabla. La media (0.356) queda al nivel del
  documento.
- **AMI: no sube de forma limpia.** Sin gigantes, la media dura (0.2004) queda algo por debajo del
  documento con Ward (0.2105). En el corpus completo queda claramente por debajo (0.373 contra
  0.452). El AMI del documento viene de reconocer tablas: su ARI~tbl es 0.87–0.96. Aun así, la
  cifra es la que es.
- **`conflicto` y `conflicto fusionado` dan la misma partición con Ward sin gigantes.** Con α=0.2 y
  ε=0.3, el documento y la tabla agregada se quedan con el 3 % y el 6.5 % de la varianza, y no
  mueven ninguna columna. En el corpus completo sí cambian algo: 0.4106 contra 0.4185.
- **Informe de salud.** La F1-macro fuera de fold pasa de 0.5238 ± 0.069 (bloque suave) a
  0.5048 ± 0.074 (bloque duro). Está dentro de la desviación, pero es algo menor. Sin bloque de
  conflicto era 0.5168 ± 0.090. El bloque no aporta al clasificador supervisado.

#### Lo que se midió y no funciona

| palanca                                                        | resultado (sin gigantes, ciego)                              |
| -------------------------------------------------------------- | ------------------------------------------------------------ |
| bloque de "forma del nombre" (reservada, genérico, inglés, mezcla, flag) | HDBSCAN colapsa a k=2 (-0.08). Ward baja a 0.10–0.12 con los dos fraseos probados |
| votar o promediar la expectativa entre fraseos y encoders      | las cuotas de voto actúan como expectativa suave y vuelve k=2. Dos votantes coinciden en el 42 % de las columnas |
| otros encoders como lector del nombre (mpnet, e5-small, e5-base) | ninguno supera a MiniLM de forma estable. mpnet da 0.04–0.06 |
| darle al lector el comentario de columna                       | la media baja de 0.126 a 0.102 (0.106 sin palabras de tipo)   |
| darle al lector "columna X de la tabla Y"                      | la media baja a 0.081 y reaparece algo de fuga (ARI~tbl 0.04) |
| revisar etiquetas (dividir `inconsistent_naming`, sacar redundantes de `clean`) | +0.015 ARI y +0.03 AMI. No se aplicó por decisión del autor |

El cuello de botella que queda es el **lector**. La lectura zero-shot de un nombre de columna desnudo
es ruidosa: los tres fraseos coinciden en la familia de una columna el 58–67 % de las veces. Cada
palanca que intentó darle más contexto o más votos empeoró el resultado.


#### Intervalos de confianza (2026-09-19): qué diferencias son reales

Con 153 columnas, una diferencia de 0.02 a 0.04 puede ser ruido. `bootstrap_compare`
(`cli experiment --bootstrap N`) puntúa cada configuración sobre los **mismos** remuestreos, así que
las diferencias son pareadas. Tiene dos modos, y los dos toman el 80 % de las columnas sin
reemplazo:

- **subsample**: vuelve a correr todo el pipeline en cada remuestreo, incluida la elección ciega de
  k o de la celda. Es la prueba exigente. Se usaron 500 remuestreos.
- **fixed**: agrupa una vez con todas las columnas y solo re-puntúa esas particiones fijas sobre
  cada remuestreo. Se usaron 2000 remuestreos.

Una primera versión del modo fixed muestreaba **con** reemplazo. Las columnas duplicadas inflan ARI
y AMI: el intervalo de AMI del documento, 0.215 a 0.343, ni siquiera contenía su propia estimación
puntual de 0.2105. Se corrigió antes de reportar nada. Un test fija que, con el 100 % de las
columnas, el modo fixed reproduce exactamente la estimación puntual.

**Sin gigantes, modo subsample** (500 remuestreos, diferencia contra documento / Ward):

| configuración               |    ARI [IC 95 %]      |    AMI [IC 95 %]      | Δ ARI [IC 95 %]        | gana | Δ AMI [IC 95 %]         | gana |
| --------------------------- | --------------------- | --------------------- | ---------------------- | ---: | ----------------------- | ---: |
| documento / Ward            | 0.051 [0.034, 0.068]  | 0.211 [0.155, 0.234]  | —                      |    — | —                       |    — |
| documento / HDBSCAN         | 0.045 [-0.001, 0.088] | 0.150 [0.068, 0.256]  | -0.003 [-0.047, +0.035] |  47 % | -0.036 [-0.129, +0.052] |  21 % |
| conflicto `orig` / Ward     | 0.169 [0.125, 0.202]  | 0.246 [0.195, 0.284]  | +0.112 [+0.069, +0.154] | 100 % | +0.041 [-0.025, +0.108] |  90 % |
| **conflicto media / Ward**  | 0.126 [0.098, 0.148]  | 0.200 [0.166, 0.233]  | **+0.073 [+0.042, +0.106]** | 100 % | +0.003 [-0.053, +0.061] | 53 % |
| conflicto media / HDBSCAN   | 0.129 [0.087, 0.153]  | 0.165 [0.126, 0.201]  | +0.069 [+0.033, +0.108] | 100 % | -0.036 [-0.091, +0.024] |  11 % |

**Sin gigantes, modo fixed, cada fraseo por separado:** los tres superan al documento en ARI con un
intervalo que excluye el cero. `orig` gana por +0.119 [+0.083, +0.158], `W2` por +0.045
[+0.017, +0.072] y `W3` por +0.061 [+0.030, +0.093]. En AMI, los tres intervalos incluyen el cero.

**Corpus completo, modo subsample:**

| configuración               | Δ ARI [IC 95 %]          | gana | Δ AMI [IC 95 %]           | gana |
| --------------------------- | ------------------------ | ---: | ------------------------- | ---: |
| conflicto `orig` / Ward     | +0.056 [+0.002, +0.123]  |  98 % | -0.076 [-0.137, -0.009]  |   1 % |
| conflicto media / Ward      | -0.010 [-0.059, +0.024]  |  32 % | -0.074 [-0.133, -0.027]  |   0 % |
| conflicto media / HDBSCAN   | -0.127 [-0.181, -0.074]  |   0 % | -0.098 [-0.145, -0.050]  |   0 % |

En el modo fixed, sobre el corpus completo, `W3` pierde en ARI por -0.059 [-0.085, -0.030]. Los
tres fraseos pierden en AMI por entre -0.07 y -0.08, con intervalos que excluyen el cero.

#### Qué se puede afirmar y qué no

- **Afirmable:** sin las tablas gigantes, la representación de conflicto con Ward mejora el ARI
  sobre el documento. La mejora sobrevive a remuestrear las columnas con el pipeline completo,
  elección ciega incluida, y a cambiar la redacción de las anclas. La estimación es +0.073, con
  intervalo [+0.042, +0.106], y gana en el 100 % de los remuestreos.
- **Afirmable:** el conflicto elimina la fuga de tabla. Su ARI~tbl es cercano a 0, contra 0.87 del
  documento.
- **No afirmable:** que mejore el AMI sin gigantes. Es un empate, [-0.053, +0.061].
- **No afirmable, y lo contrario es cierto:** que iguale al documento en el corpus completo. Ahí el
  documento es mejor en AMI por unos 0.07, con intervalo que excluye el cero, y empata en ARI con la
  media. La ventaja de `orig` en ARI es marginal: su intervalo empieza en +0.002. El documento gana
  en el corpus completo reconociendo las dos tablas gigantes, con ARI~tbl 0.96.
- **Ward frente a HDBSCAN para el documento:** no hay diferencia. Para el conflicto, Ward es igual
  sin gigantes y claramente mejor con el corpus completo.


#### Anclas hechas de nombres de ejemplo (2026-09-19): resultado negativo

Hipótesis: el lector compara nombres de dos palabras contra frases completas ("este campo guarda
una fecha, un momento…"). Si cada familia se describe con nombres de columna de ejemplo, el lector
compararía nombres contra nombres y leería mejor.

Se escribieron tres conjuntos independientes de 7 u 8 nombres inventados por familia. E1 y E2 están
en español; E3 mezcla inglés y español. Se descartó cualquier ejemplo idéntico a un nombre de
columna del esquema: `es_activo` en E2, y `monto`, `nombre` y `activo` en E3. Cada familia se
representó de dos maneras: por el centroide de sus ejemplos, o por el ejemplo más cercano a la
columna. Se usaron expectativa dura, Ward con k por silueta y el fusionado, igual que el candidato #7.

| anclas, media de 3 conjuntos | sin gigantes ARI / AMI | completo ARI / AMI | acuerdo del lector entre conjuntos, sin gigantes / completo |
| ---------------------------- | ---------------------: | -----------------: | ---------------------------------------------------------: |
| frases (candidato #7)        |        0.1256 / 0.2004 |    0.3561 / 0.3729 |                                                0.63 / 0.53 |
| centroide de ejemplos        |        0.1392 / 0.1849 |    0.2704 / 0.3341 |                                                0.77 / 0.64 |
| ejemplo más cercano          |        0.0985 / 0.1809 |    0.2827 / 0.3977 |                                                0.69 / 0.62 |

Bootstrap pareado, 300 submuestras del 80 %, centroide menos frases:

| corpus       | Δ ARI [IC 95 %]          | gana | Δ AMI [IC 95 %]          | gana |
| ------------ | ------------------------ | ---: | ------------------------ | ---: |
| sin gigantes | +0.016 [-0.008, +0.043]  |  90 % | -0.008 [-0.033, +0.016] |  27 % |
| completo     | -0.079 [-0.128, -0.032]  |   0 % | -0.014 [-0.063, +0.036] |  36 % |

Lectura:

- **No se adopta.** Sin gigantes empata. En el corpus completo pierde en ARI, con un intervalo que
  excluye el cero.
- **Un lector más estable no es un lector más acertado.** Con centroides, los tres conjuntos
  coinciden más entre sí: 0.77 frente a 0.63 sin gigantes. Pero el agrupamiento no mejora, así que
  el acuerdo entre redacciones no era la medida del problema. Los tres conjuntos de ejemplos pueden
  coincidir en leer mal la misma columna.
- La siguiente palanca de la lista es cambiar la forma de leer, no las anclas: un clasificador de
  inferencia (NLI) multilingüe en lugar del coseno. Requiere descargar un modelo.


#### Lector por inferencia (NLI) en lugar del coseno (2026-09-19): resultado negativo

Hipótesis: un modelo entrenado para decidir si una frase se sigue de otra leería mejor el nombre
que un coseno entre vectores. La premisa es el nombre de la columna y cada familia es una hipótesis.
La puntuación de implicación reemplaza al coseno. El resto no cambia: expectativa dura con margen,
Ward con k por silueta, el fusionado, y las tres redacciones de las frases como hipótesis.

Se probaron tres modelos multilingües con licencia MIT, todos de la familia BERT:
`Recognai/bert-base-spanish-wwm-cased-xnli` (BERT en español), `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`
y `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`. Premisa: "Nombre de la columna: {nombre}."

| lector, media de 3 redacciones | sin gigantes ARI / AMI | completo ARI / AMI | acuerdo entre redacciones, sin gigantes |
| ------------------------------ | ---------------------: | -----------------: | -------------------------------------: |
| **coseno MiniLM (actual)**     |    **0.1256 / 0.2004** | **0.3561 / 0.3729** |                                   0.63 |
| NLI BERT español               |       -0.0284 / 0.0838 |    0.0482 / 0.1181 |                                   0.41 |
| NLI MiniLMv2 multilingüe       |        0.0282 / 0.1193 |    0.1504 / 0.2001 |                                   0.77 |
| NLI mDeBERTa multilingüe       |        0.0471 / 0.1503 |    0.1894 / 0.2658 |                                   0.23 |

La diferencia es demasiado grande para no dudar del montaje, así que se revisó sin etiquetas. Se
escribieron 18 nombres inequívocos con su familia obvia, como "fecha nacimiento" para fecha,
"salario" para importe, "activo" para booleano o "imagen" para binario. Se probaron tres plantillas
de premisa e hipótesis, incluida la estándar de zero-shot, "Este ejemplo es {familia}.":

| lector                    | aciertos de 18, según plantilla |
| ------------------------- | ------------------------------: |
| coseno MiniLM (actual)    |                              18 |
| NLI BERT español          |                        8, 9, 10 |
| NLI MiniLMv2 multilingüe  |                       8, 8, 10  |
| NLI mDeBERTa multilingüe  |                      8, 10, 11  |

Lectura:

- **No se adopta.** Los modelos NLI fallan en los casos fáciles, no solo en los ambiguos. Cambiar
  la plantilla no lo arregla. Un nombre de columna desnudo no es una frase que implique algo, que es
  lo que estos modelos aprendieron a juzgar.
- **El coseno acierta todos los casos obvios.** Lo que le queda al lector actual son los nombres
  ambiguos de verdad, como `MONTO` (fecha o importe), `NIVEL` o `INTENTOS_FALLIDOS`. Ninguna
  palanca probada sobre el lector (redacción, votación, ejemplos, contexto, otros encoders, NLI)
  mejora esa parte.
- La lista de 18 nombres la escribió quien hizo el experimento. Es una prueba de validez aparente,
  no una medida. Sirve para descartar un fallo del montaje, no para puntuar lectores.


#### Señal nueva para `inconsistent_naming`, `polymorphic`, `eav` y `reserved_words` (2026-09-19): resultado negativo, con diagnóstico

El bloque de conflicto no puede ver estas cuatro clases por diseño: suman 38 de las 153 columnas sin
gigantes. Se buscó una señal nueva hecha con BERT (MiniLM), sin etiquetas y sin reglas.

**Qué son estas clases en el corpus.** Tres de las cuatro son propiedades de la tabla o de los
hermanos, no de la columna:

| clase                 | dónde está                                                                  |
| --------------------- | --------------------------------------------------------------------------- |
| `eav`                 | `CONFIGURACION`, 6 de 6 columnas, `ID` incluido                             |
| `inconsistent_naming` | `TBL_DATOS` 9 de 9, `ORDENES_COMPRA` 8 de 11, cuatro `FLAG_*` de `TODO_EN_UNO` |
| `reserved_words`      | `REPORTES`, 4 de 6                                                          |
| `polymorphic`         | nombres con "o" (`FECHA_O_DIRECCION`) y discriminadores (`TIPO`, `TIPO_REGISTRO`) |

Solo 2 de las 38 columnas son indistinguibles por construcción: `CONFIGURACION.ID` y
`CONFIGURACION.ACTIVO`. Tienen un gemelo exacto en otra tabla, con el mismo nombre, tipo y
restricciones, pero con otra etiqueta.

**Cinco señales, todas sobre el nombre desnudo:**

1. **vacío semántico**: 1 menos el mejor coseno contra las 6 familias.
2. **divergencia interna**: la mayor distancia entre las palabras del propio nombre.
3. **atípico entre hermanos**: la distancia al centroide de los demás nombres de la tabla.
4. **choque de convención**: el mayor coseno con un hermano de escritura distinta.
5. **heterogeneidad de la tabla**: la distancia media entre los nombres de la tabla.

Separan poco. Por etiqueta, el vacío semántico vale 0.65–0.69 en `inconsistent_naming` y
`reserved_words`, contra 0.58 en `clean`. El choque vale 0.75 en `inconsistent_naming`, contra 0.62
en `clean`. La divergencia es mayor en `impossible_data` (0.35) que en `polymorphic` (0.26).

**Añadidas al fusionado** (Ward ciego, media de 3 redacciones, sin gigantes; "pares juntos" es la
fracción de pares de la clase que caen en el mismo cluster):

| configuración                       |    ARI |    AMI | ARI~tbl | pares `eav` | pares `reserved` |
| ----------------------------------- | -----: | -----: | ------: | ----------: | ---------------: |
| base                                | 0.1256 | 0.2004 |   0.002 |        0.02 |             0.28 |
| + las 5 señales, peso 0.35          | 0.1291 | 0.2080 |   0.002 |        0.02 |             0.28 |
| + las 5 señales, peso 2.0           | 0.0412 | 0.1565 |   0.074 |        0.11 |             0.50 |
| solo las 5 señales                  | 0.0349 |      — |       — |        0.40 |             0.50 |
| + medias por tabla de 1–4, peso 1.0 | 0.0575 | 0.1609 |   0.112 |        0.69 |             0.83 |

Lectura:

- **No se adopta ninguna.** Con peso chico no mueven las columnas objetivo. Con peso grande agrupan
  mejor las clases objetivo, pero rompen los clusters de conflicto y el ARI total cae a 0.04–0.06.
- **El problema es de granularidad, no de señal.** Los conflictos de tipo agrupan columnas de muchas
  tablas: su ARI~tbl es cercano a 0. Estas cuatro clases agrupan columnas de una misma tabla. Un
  solo clustering plano de columnas no puede servir a las dos cosas a la vez. La medida por tabla lo
  muestra: al subir su peso, `eav` y `reserved_words` se juntan, del 2 % al 69 % y del 28 % al 83 %,
  pero todo lo demás se desordena.
- **Esto explica también `giant_table`**, la otra clase que es propiedad de la tabla, y por qué el
  documento gana en el corpus completo: reconoce tablas (ARI~tbl 0.96), y en estas clases reconocer
  la tabla es acertar la etiqueta.
- **La vía que queda es un diseño en dos niveles.** Un clustering de columnas por conflicto para los
  anti-patrones de columna, y un clustering de tablas, con señales de BERT sobre el conjunto de
  nombres de cada tabla, para los anti-patrones de tabla: `giant_table`, `eav`, `inconsistent_naming`
  de tabla entera y `reserved_words`. Requiere decidir cómo se etiqueta y se puntúa una tabla, y solo
  hay 23 tablas, así que la evaluación de ese nivel tendrá poca potencia estadística.


### Candidato #8 (2026-09-19): diseño en dos niveles — columnas y tablas

La sección anterior mostró que un solo clustering de columnas no puede servir a la vez a los
anti-patrones de columna, que cruzan tablas, y a los de tabla, que viven dentro de una. Este
candidato separa los dos niveles.

#### Diseño

- **Nivel de columna.** Es el fusionado de conflicto con Ward y k por silueta, sin cambios.
- **Nivel de tabla.** Cada tabla se describe con 6 medidas de BERT (MiniLM) sobre los nombres de sus
  columnas: vacío semántico medio, divergencia interna media, atípico medio, choque máximo,
  heterogeneidad, y proporción de columnas cuyo nombre contradice su tipo
  (`features/name_signals.py`). Una tabla es anómala si su distancia a la tabla mediana supera la
  valla de Tukey, Q3 + 1.5·IQR. Las tablas anómalas se agrupan entre sí por enlace simple, cortado
  en esa misma valla (`clustering/table_level.py`). Nada ahí ve una etiqueta.
- **Combinación.** Una columna de una tabla normal conserva su cluster de columna. Una columna de
  una tabla anómala toma el grupo de su tabla, porque el anti-patrón es de la tabla.
- **Etiquetas de tabla, solo para puntuar el nivel de tabla.** Se derivan de las etiquetas de
  columna: una tabla recibe la etiqueta no-clean más frecuente si cubre al menos la mitad de sus
  columnas. Salen 2 `giant_table`, 2 `inconsistent_naming`, 1 `eav`, 1 `reserved_words` y 17
  tablas sin anti-patrón de tabla. No se escribió ninguna etiqueta nueva.

`cli experiment --two-level` lo reporta, y `--bootstrap` lo incluye cuando el pickle trae
`e_name`/`name_intrinsic`.

#### La regla del nivel de tabla, y cuántas se probaron

Se probaron cuatro variantes, en este orden. Se declara porque es una elección hecha mirando
resultados:

1. **Ward sobre tablas, con k por silueta, y el cluster mayor como "normal".** Falló sin gigantes:
   la silueta eligió 6 o 7 clusters y 11 de 21 tablas quedaron como anómalas. ARI 0.07.
2. **Valla de Tukey, cada tabla anómala en su propio grupo.** Separó las dos tablas gigantes, que
   son el mismo anti-patrón. ARI completo 0.393.
3. **Valla de Tukey, con las anómalas unidas por enlace simple cortado en la valla.** Es la regla
   adoptada.
4. **Exclusión de tablas con menos de 3 columnas** (`MIN_TABLE_COLUMNS`). Se añadió al ver el
   bootstrap. Una tabla sin hermanos tiene señales de hermanos iguales a cero por construcción, un
   valor extremo que es un artefacto. Antes de esta regla, 23 de las tablas marcadas en las
   submuestras tenían 1 o 2 columnas, y las dos gigantes quedaban juntas en 31 de 60 submuestras.
   Después, en 44 de 60. **En el corpus completo no cambia ninguna cifra**, porque ninguna tabla
   tiene menos de 3 columnas.

#### Resultados

Sobre los datos completos, media de 3 redacciones, todo elegido a ciegas:

| corpus       | configuración        |    ARI |    AMI | ARI~tbl | tablas marcadas                    |
| ------------ | -------------------- | -----: | -----: | ------: | ---------------------------------- |
| sin gigantes | documento            | 0.0505 | 0.2105 |   0.867 | —                                  |
| sin gigantes | conflicto plano      | 0.1256 | 0.2004 |   0.002 | —                                  |
| sin gigantes | **dos niveles**      | 0.1416 | 0.2447 |   0.037 | `TBL_DATOS`                        |
| completo     | documento            | 0.3659 | 0.4522 |   0.963 | —                                  |
| completo     | conflicto plano      | 0.3561 | 0.3729 |   0.380 | —                                  |
| completo     | **dos niveles**      | 0.6479 | 0.5020 |   0.468 | `BACKUP_DATOS` + `TABLA_BASE_DATOS` |

Las tres redacciones marcan las mismas tablas. El nivel de tabla, puntuado contra las etiquetas de
tabla derivadas, da ARI 0.305 sin gigantes y 0.386 en el corpus completo.

**Bootstrap pareado** (500 submuestras del 80 %, pipeline ciego completo en cada una), dos niveles
menos conflicto plano:

| corpus       | Δ ARI [IC 95 %]          | gana | Δ AMI [IC 95 %]          | gana |
| ------------ | ------------------------ | ---: | ------------------------ | ---: |
| sin gigantes | +0.004 [-0.012, +0.020]  |  50 % | +0.014 [-0.013, +0.051] |  54 % |
| completo     | +0.169 [-0.031, +0.328]  |  83 % | +0.086 [+0.000, +0.169] |  90 % |

Contra el documento, en el corpus completo: ARI +0.158 [-0.055, +0.300], con el 85 % de victorias,
y AMI +0.012 [-0.099, +0.088].

#### Lectura

- **Sin gigantes, el nivel de tabla no aporta nada demostrable.** La subida de 0.126 a 0.142 sobre
  los datos completos depende de marcar `TBL_DATOS`, y eso solo ocurre en parte de las submuestras.
  Con las otras tablas objetivo (`CONFIGURACION`, `ORDENES_COMPRA`, `REPORTES`) las señales de
  nombre no alcanzan: nunca cruzan la valla.
- **En el corpus completo la mejora es grande cuando ocurre, pero no es significativa al 95 %.** Es
  bimodal: si las dos tablas gigantes caen en el mismo grupo, el ARI salta a ~0.65; si no, se queda
  en el nivel del conflicto plano. Gana en el 83 % de las submuestras en ARI y en el 90 % en AMI, y
  el intervalo de AMI toca el cero exactamente. La mejora es probable, no demostrada.
- **Lo que sí aporta con claridad es interpretabilidad.** El diseño dice qué tablas tienen un
  problema de tabla, cosa que el clustering plano no puede decir. Y lo hace con una regla sin
  parámetros que ajustar.
- **Decisión:** queda como opción (`algorithm="two-level"`, `--two-level`), no como comportamiento
  por defecto. Para afirmar una mejora sin gigantes haría falta una señal de tabla que detecte `eav`
  y la inconsistencia de nombres de `ORDENES_COMPRA`. Ninguna de las probadas lo hace.

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
