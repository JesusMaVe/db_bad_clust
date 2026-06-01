# Selección de Algoritmos de Clustering para la Optimización Automatizada de Esquemas de Bases de Datos

> Un enfoque basado en embeddings semánticos y análisis de densidad

---

## 1. Resumen

El presente reporte aborda la optimización automatizada de esquemas de bases de datos mediante aprendizaje automático no supervisado. Se propone un pipeline que combina embeddings semánticos generados por BERT con metadatos estructurales codificados numéricamente, construyendo un vector de features compuesto que representa fielmente cada atributo de una base de datos. Sobre esta representación se aplican y comparan K-Means, DBSCAN y Mean Shift, con el objetivo de agrupar atributos semánticamente relacionados y detectar anomalías de diseño.

Se desarrollan los fundamentos matemáticos de cada algoritmo, la construcción del vector híbrido, un protocolo experimental con métricas cuantitativas (Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index), y se argumenta por qué BERT + DBSCAN constituye la hipótesis principal, sin descartar los demás como líneas base.

---

## 2. Introducción y Planteamiento del Problema

### 2.1 Contexto

El diseño de bases de datos relacionales depende del criterio del arquitecto de datos. A medida que los sistemas crecen, el esquema tiende a degradarse: redundancias semánticas, inconsistencias de tipos, tablas que mezclan dominios no relacionados, y oportunidades de normalización no detectadas. La pregunta central es: ¿es posible construir un modelo de ML que, al analizar los metadatos de una BD, identifique automáticamente agrupaciones lógicas y anomalías de diseño?

### 2.2 Formulación Formal del Problema

Sea **D** una base de datos relacional con tablas **T = {t₁, t₂, ..., tₘ}**, donde cada tabla tᵢ contiene atributos **Aᵢ = {aᵢ₁, aᵢ₂, ..., aᵢₙᵢ}**. El universo completo de atributos es **A = A₁ ∪ A₂ ∪ ... ∪ Aₘ**, con |A| = N.

Cada atributo aⱼ ∈ A posee metadatos Mⱼ:

| Metadato | Descripción              | Tipo       |
| -------- | ------------------------ | ---------- |
| nⱼ       | Nombre textual           | String     |
| τⱼ       | Tipo de dato             | Categórico |
| pkⱼ      | Llave primaria           | {0, 1}     |
| fkⱼ      | Llave foránea            | {0, 1}     |
| nullⱼ    | Admite nulos             | {0, 1}     |
| idxⱼ     | Tiene índice             | {0, 1}     |
| tⱼ       | Tabla a la que pertenece | Referencia |

El objetivo es construir una función de representación **φ: A → ℝᵈ** y aplicar un algoritmo de clustering:

```
C: {φ(a₁), ..., φ(aₙ)} → {G₁, G₂, ..., Gₖ, G_ruido}
```

### 2.3 Hipótesis

**H₁ (principal):** La combinación de embeddings BERT con features estructurales, agrupada con DBSCAN, produce clusters más coherentes y detecta más anomalías que K-Means o Mean Shift sobre la misma representación.

**H₀ (nula):** No existe diferencia significativa en la calidad de clusters entre los tres algoritmos sobre el vector compuesto.

---

## 3. Marco Teórico y Fundamentos Matemáticos

### 3.1 BERT: Representación Semántica de Texto

BERT (Bidirectional Encoder Representations from Transformers) no es un algoritmo de clustering — es un modelo de **embeddings** que transforma texto en vectores densos de ℝ⁷⁶⁸. Su relevancia radica en capturar relaciones semánticas profundas: entiende que `customer_name` y `nombre_cliente` son conceptualmente equivalentes, algo imposible con TF-IDF o one-hot encoding.

#### 3.1.1 Arquitectura del Transformer Encoder

El mecanismo central es la **Self-Attention**. Dados vectores de entrada X = [x₁, ..., xₙ], se computan tres proyecciones:

```
Q = X·Wq    (Queries)
K = X·Wk    (Keys)
V = X·Wv    (Values)
```

donde **Wq, Wk ∈ ℝᵈˣᵈᵏ** y **Wv ∈ ℝᵈˣᵈᵛ** son parámetros aprendidos. La función de atención escalada:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) \cdot V$$

El factor **√dₖ** previene que los productos punto saturen el softmax y produzcan gradientes nulos.

#### 3.1.2 Multi-Head Attention

BERT emplea h = 12 cabezas en paralelo, cada una con sus propias proyecciones, para capturar diferentes tipos de relaciones simultáneamente:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) \cdot W^O$$

$$\text{donde} \quad \text{head}_i = \text{Attention}(QW_i^Q,\; KW_i^K,\; VW_i^V)$$

Para BERT-base: h = 12, dₖ = dᵥ = 64, dimensión total = 768.

_(Insertar Diagrama 4 — Self-Attention de BERT)_

#### 3.1.3 Generación del Embedding de Atributo

```
e_texto(aⱼ) = BERT([CLS] + tokenize(nⱼ) + [SEP])[CLS]  ∈ ℝ⁷⁶⁸
```

El vector del token `[CLS]` de la última capa actúa como representación agregada de toda la secuencia.

---

### 3.2 K-Means: Clustering Basado en Centroides

#### 3.2.1 Formulación Matemática

K-Means divide N observaciones en exactamente k clusters S = {S₁, ..., Sₖ}, minimizando la **inercia (WCSS)**:

$$J(S) = \sum_{i=1}^{k} \sum_{x \in S_i} \|x - \mu_i\|^2$$

donde **μᵢ = (1/|Sᵢ|) · Σₓ∈Sᵢ x** es el centroide del cluster Sᵢ.

#### 3.2.2 Algoritmo de Lloyd

Itera entre dos pasos hasta convergencia:

**E-step (asignación):**

$$S_i^{(t)} = \left\{ x_j : \|x_j - \mu_i^{(t)}\|^2 \leq \|x_j - \mu_l^{(t)}\|^2 \;\; \forall l \right\}$$

**M-step (actualización):**

$$\mu_i^{(t+1)} = \frac{1}{|S_i^{(t)}|} \sum_{x \in S_i^{(t)}} x$$

Cada iteración reduce monótonamente J(S), garantizando convergencia a un mínimo local.

#### 3.2.3 Complejidad Computacional

O(Nkd) por iteración, donde N = puntos, k = clusters, d = dimensionalidad.

#### 3.2.4 Limitaciones para el Problema

- Requiere k a priori (desconocido en una BD arbitraria).
- Asume clusters esféricos e isótropos.
- No detecta outliers: todo punto se asigna forzosamente a algún cluster.

---

### 3.3 DBSCAN: Clustering Basado en Densidad

#### 3.3.1 Definiciones Fundamentales

DBSCAN se parametriza con **ε** (radio de vecindad) y **MinPts** (mínimo de puntos para región densa).

| Concepto                    | Definición formal                                                                     |
| --------------------------- | ------------------------------------------------------------------------------------- |
| **ε-vecindad**              | Nε(p) = {q ∈ D : dist(p, q) ≤ ε}                                                      |
| **Punto núcleo**            | p es núcleo si \|Nε(p)\| ≥ MinPts                                                     |
| **Alcanzable por densidad** | Existe cadena p₁=p, p₂, ..., pₙ=q donde cada pᵢ₊₁ es directamente alcanzable desde pᵢ |
| **Densamente conectado**    | Existe o tal que p y q son ambos alcanzables desde o                                  |

#### 3.3.2 Propiedad Formal de Clustering

Un cluster C satisface:

- **Conexión:** ∀ p, q ∈ C → p y q están densamente conectados.
- **Maximalidad:** Si p ∈ C y q es alcanzable desde p → q ∈ C.

Los puntos que no pertenecen a ningún cluster son **ruido** — la propiedad más valiosa para detectar anomalías de diseño.

#### 3.3.3 Función de Distancia

Distancia euclidiana estándar:

$$\text{dist}(p, q) = \sqrt{\sum_{i=1}^{d}(p_i - q_i)^2}$$

Para embeddings BERT (d = 768), la **distancia coseno** es más apropiada:

$$\text{dist\_cos}(p, q) = 1 - \frac{p \cdot q}{\|p\| \cdot \|q\|}$$

Normaliza magnitudes y se enfoca en la orientación angular, más relevante para semántica.

#### 3.3.4 Selección de Hiperparámetros

**ε:** Método del k-distance graph. Se calcula la distancia al k-ésimo vecino más cercano (k = MinPts) para cada punto, se ordenan ascendentemente y se identifica el "codo" (máxima curvatura).

**MinPts:** Regla heurística: MinPts ≥ d + 1, donde d es la dimensionalidad post-reducción.

#### 3.3.5 Complejidad Computacional

- Con KD-tree o ball tree: **O(N log N)**
- Sin indexación: O(N²)

---

### 3.4 Mean Shift: Clustering por Desplazamiento de Media

#### 3.4.1 Estimación de Densidad por Kernel (KDE)

$$\hat{f}(x) = \frac{1}{Nh^d} \sum_{i=1}^{N} K\!\left(\frac{x - x_i}{h}\right)$$

Con kernel gaussiano: **K(x) = (2π)^(−d/2) · exp(−‖x‖²/2)**. El parámetro h (bandwidth) controla la suavidad de la estimación.

#### 3.4.2 El Vector de Mean Shift

$$m(x) = \frac{\sum_{i=1}^{N} x_i \cdot K\!\left(\frac{x - x_i}{h}\right)}{\sum_{i=1}^{N} K\!\left(\frac{x - x_i}{h}\right)} - x$$

El procedimiento iterativo: **x^(t+1) = x^(t) + m(x^(t))** desplaza cada punto hacia el modo local hasta convergencia. Los puntos que convergen al mismo modo forman un cluster.

#### 3.4.3 Limitaciones para el Problema

- Complejidad **O(TN²)** por iteración — prohibitivo para esquemas grandes.
- En alta dimensionalidad (d = 768+), el volumen del espacio crece exponencialmente (curse of dimensionality), degradando la estimación KDE hasta perder significancia estadística.

---

## 4. Diseño del Vector de Features Compuesto

_(Insertar Diagrama 2 — Construcción del Vector de Features)_

### 4.1 Componente Semántico: Embedding BERT

```
e_texto(aⱼ) = BERT([CLS] + tokenize(nⱼ) + [SEP])[CLS]  ∈ ℝ⁷⁶⁸
```

Preprocesamiento previo:

1. Separar camelCase / snake_case → `fechaNacimiento` → `fecha nacimiento`
2. Expandir abreviaturas → `fk` → `foreign key`, `dt` → `date`
3. Agregar contexto de tabla → `"clientes: nombre"`

### 4.2 Componente Estructural: Features Categóricos y Binarios

**Tipo de dato — One-Hot Encoding:**

Sea T = {INT, BIGINT, FLOAT, VARCHAR, TEXT, DATE, DATETIME, BOOLEAN, ...} con |T| = p:

```
e_tipo(aⱼ) = [0, ..., 1, ..., 0]ᵀ  ∈ ℝᵖ
```

**Restricciones — Codificación binaria:**

```
e_rest(aⱼ) = [pkⱼ, fkⱼ, uniqueⱼ, nullⱼ, idxⱼ, auto_incⱼ]ᵀ  ∈ {0,1}⁶
```

**Features estadísticos (opcionales):**

Si se dispone de datos reales: cardinalidad normalizada, proporción de nulos, entropía de Shannon **H(aⱼ) = −Σᵥ p(v) log₂ p(v)**, estadísticos de longitud. Se agrupan en **e_est(aⱼ) ∈ ℝˢ**.

### 4.3 Construcción del Vector Compuesto

Normalización Z-score por componente: **ê = (e − μₑ) / σₑ**

Concatenación ponderada:

$$\varphi(a_j) = \bigl[\alpha \cdot \hat{e}_{\text{texto}} \;\oplus\; \beta \cdot \hat{e}_{\text{tipo}} \;\oplus\; \gamma \cdot \hat{e}_{\text{rest}} \;\oplus\; \delta \cdot \hat{e}_{\text{est}}\bigr]$$

Pesos iniciales sugeridos (hiperparámetros a optimizar):

| Peso | Componente       | Valor inicial |
| ---- | ---------------- | ------------- |
| α    | Semántico (BERT) | 0.60          |
| β    | Tipo de dato     | 0.15          |
| γ    | Restricciones    | 0.15          |
| δ    | Estadísticos     | 0.10          |

Restricción: **α + β + γ + δ = 1**

Dimensionalidad resultante: **d = 768 + p + 6 + s ≈ 800**

### 4.4 Reducción de Dimensionalidad

Para algoritmos sensibles a la dimensionalidad, especialmente Mean Shift:

**PCA:** Proyección lineal a k componentes que explican ≥ 95% de varianza acumulada.

**UMAP:** Preserva estructura topológica local en proyección no lineal. Parámetros de partida: `n_neighbors=15`, `min_dist=0.1`. Preferido sobre PCA para datos de alta dimensionalidad, ya que preserva mejor las relaciones de vecindad críticas para algoritmos de densidad.

---

## 5. Análisis Comparativo de Algoritmos

_(Insertar Diagrama 3 — Comparación Visual)_

| Propiedad             | K-Means                           | DBSCAN                          | Mean Shift                  |
| --------------------- | --------------------------------- | ------------------------------- | --------------------------- |
| Tipo                  | Centroides (particionamiento)     | Densidad                        | Modos de densidad           |
| Requiere k            | Sí                                | No                              | No                          |
| Detección de outliers | No (todo se asigna)               | **Sí (etiqueta nativa)**        | No                          |
| Forma de clusters     | Esférica (Voronoi)                | Arbitraria                      | Arbitraria                  |
| Complejidad           | O(Nkd) / iter                     | O(N log N)                      | O(TN²) / iter               |
| Alta dimensionalidad  | Sensible                          | Moderada                        | Muy sensible                |
| Hiperparámetros       | k, inicialización                 | ε, MinPts                       | bandwidth h                 |
| Interpretación en BD  | Grupos forzados de tamaño similar | Dominios semánticos + anomalías | Picos de densidad semántica |

DBSCAN ofrece la combinación más favorable: descubrimiento automático de k, detección nativa de ruido interpretable como anomalías de diseño, clusters de forma arbitraria, y complejidad manejable.

---

## 6. Metodología Experimental

_(Insertar Diagrama 5 — Metodología Experimental)_

### 6.1 Diseño del Experimento

Diseño comparativo de tres tratamientos sobre la misma representación vectorial: **9 combinaciones** (3 algoritmos × 3 configuraciones de dimensionalidad).

**Fase 1 — Recopilación de Datos**

Corpus de esquemas de diferentes dominios y complejidades:

- Bases de datos públicas: Sakila (MySQL), AdventureWorks, Chinook, Northwind.
- Esquemas sintéticos con anomalías intencionalmente introducidas.
- Esquemas reales anonimizados (si disponibles).

**Fase 2 — Preprocesamiento y Embeddings**

Pipeline de preprocesamiento (Sección 4.1) + generación de embeddings con `bert-base-multilingual-cased` (Hugging Face) + construcción de φ(aⱼ) con pesos iniciales.

**Fase 3 — Reducción de Dimensionalidad**

Tres configuraciones experimentales:

1. Vector completo sin reducción (d ≈ 800)
2. PCA a 50 componentes (varianza acumulada ≥ 95%)
3. UMAP a 20 dimensiones (`n_neighbors=15`, `min_dist=0.1`)

**Fase 4 — Aplicación de Algoritmos**

- **K-Means:** k ∈ [2, √N], método del codo + silueta, 10 inicializaciones K-Means++.
- **DBSCAN:** ε vía k-distance graph, MinPts ∈ [d+1, 2(d+1)], métrica coseno.
- **Mean Shift:** bandwidth estimado con `estimate_bandwidth(quantile=0.3)`.

**Fase 5 — Evaluación y Análisis**

Métricas cuantitativas + análisis cualitativo de coherencia semántica.

### 6.2 Métricas de Evaluación

#### Silhouette Score

Para cada punto i, a(i) = distancia media al resto de su cluster, b(i) = distancia media mínima a cualquier otro cluster:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i),\; b(i))}$$

**s(i) ∈ [−1, 1].** Valor global: S = (1/N) Σᵢ s(i). Mayor es mejor.

#### Davies-Bouldin Index

$$\text{DB} = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \frac{\sigma_i + \sigma_j}{d(c_i,\, c_j)}$$

σᵢ = dispersión promedio del cluster i, d(cᵢ, cⱼ) = distancia entre centroides. **Menor es mejor.**

#### Calinski-Harabasz Index

$$\text{CH} = \frac{\text{tr}(B_k)\,/\,(k-1)}{\text{tr}(W_k)\,/\,(N-k)}$$

Donde:

- **Bₖ = Σᵢ nᵢ(cᵢ − c)(cᵢ − c)ᵀ** → dispersión inter-cluster
- **Wₖ = Σᵢ Σₓ∈Cᵢ (x − cᵢ)(x − cᵢ)ᵀ** → dispersión intra-cluster

**Mayor es mejor** (alta separación inter, baja dispersión intra).

#### Métricas Específicas del Dominio

| Métrica                    | Definición                                                            |
| -------------------------- | --------------------------------------------------------------------- |
| Coherencia de Tipo         | CT(Cᵢ) = \|tipo_mayoritario(Cᵢ)\| / \|Cᵢ\|                            |
| Tasa Detección Anomalías   | Anomalías introducidas correctamente etiquetadas como ruido           |
| Concordancia Normalización | Comparación con agrupaciones de dependencias funcionales (3FN / BCNF) |

---

## 7. Arquitectura del Pipeline Propuesto

_(Insertar Diagrama 1 — Pipeline del Sistema)_

### 7.1 Etapas

**Etapa 1 — Extracción de Metadatos:** Conexión al SGBD vía `information_schema`. Extrae nombres de tablas/columnas, tipos, restricciones, índices, relaciones FK y comentarios.

**Etapa 2 — Preprocesamiento Textual:** Separación camelCase/snake*case, expansión de abreviaturas, eliminación de prefijos redundantes (`tbl*`, `col*`, `fld*`), contextualización con nombre de tabla.

**Etapa 3 — Embeddings BERT:** Inferencia con `bert-base-multilingual-cased`. Extracción del vector `[CLS]` de la última capa: `e_texto(aⱼ) ∈ ℝ⁷⁶⁸`.

**Etapa 4 — Codificación Estructural:** One-hot de tipos, binario de restricciones, estadísticos opcionales.

**Etapa 5 — Vector Compuesto:** Normalización Z-score, ponderación α/β/γ/δ, concatenación en φ(aⱼ). Reducción UMAP opcional.

**Etapa 6 — Clustering:** DBSCAN como predeterminado; K-Means y Mean Shift como comparativas. Evaluación de métricas y selección de hiperparámetros.

**Etapa 7 — Recomendaciones:**

| Tipo                         | Señal en los clusters                                     |
| ---------------------------- | --------------------------------------------------------- |
| Redundancias semánticas      | Nombres similares en tablas diferentes → mismo cluster    |
| Sugerencias de normalización | Atributos de dominios distintos → misma tabla             |
| Inconsistencias de tipo      | Atributos semánticamente equivalentes con tipos distintos |
| Índices faltantes            | Agrupados con campos indexados pero sin índice propio     |
| Atributos anómalos           | Etiquetados como ruido por DBSCAN                         |

### 7.2 Pseudocódigo

```python
# Entrada: connection_string
# Salida: clusters, recomendaciones, métricas

metadata   = extract_schema(connection_string)
names      = preprocess_names(metadata.column_names)
e_text     = bert_encode(names)                          # ℝ^(N×768)
e_type     = one_hot_encode(metadata.data_types)         # ℝ^(N×p)
e_rest     = binary_encode(metadata.constraints)         # ℝ^(N×6)
e_stat     = compute_statistics(metadata)                # ℝ^(N×s) — opcional

phi        = weighted_concat(normalize(e_text, e_type, e_rest, e_stat), α, β, γ, δ)
phi_red    = UMAP(phi, n_components=20)                  # opcional

eps        = estimate_eps_kneedle(phi_red, MinPts)
labels     = DBSCAN(phi_red, eps=eps, min_samples=MinPts, metric='cosine')

metrics    = evaluate(phi_red, labels)                   # S, DB, CH + dominio
recs       = interpret_clusters(labels, metadata)

return clusters, recs, metrics
```

---

## 8. Stack Tecnológico

| Componente     | Tecnología                                                                           |
| -------------- | ------------------------------------------------------------------------------------ |
| Embeddings     | Hugging Face Transformers (`bert-base-multilingual-cased`)                           |
| Clustering     | scikit-learn (`KMeans`, `DBSCAN`, `MeanShift`)                                       |
| Reducción dim. | `umap-learn`, scikit-learn (`PCA`)                                                   |
| Evaluación     | scikit-learn (`silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score`) |
| Procesamiento  | Python 3.10+, NumPy, Pandas                                                          |
| Visualización  | Matplotlib, Seaborn, Plotly                                                          |
| Conexión BD    | SQLAlchemy (abstracción multi-SGBD)                                                  |

---

## 9. Conclusiones

**Hipótesis principal.** BERT + DBSCAN es la combinación más prometedora: BERT captura la semántica más rica disponible para texto corto, y DBSCAN descubre automáticamente el número de clusters sin suposiciones sobre su forma, con detección nativa de ruido directamente interpretable como anomalías de diseño.

**Valor del enfoque comparativo.** K-Means y Mean Shift aportan líneas base para cuantificar la ventaja diferencial, validación cruzada (si los tres coinciden en ciertos clusters, la evidencia es más fuerte), y análisis de sensibilidad ante cambios en hiperparámetros y representaciones.

**Trabajo futuro:** explorar Sentence-BERT (SBERT) optimizado para similitud semántica; incorporar información de grafos (relaciones FK) mediante Graph Neural Networks; extender a esquemas NoSQL (MongoDB, Cassandra).

---

## 10. Referencias

1. Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. _NAACL-HLT 2019_, 4171–4186.
2. Vaswani, A. et al. (2017). Attention Is All You Need. _NeurIPS 30_.
3. Ester, M. et al. (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise. _KDD-96_, 226–231.
4. Lloyd, S. P. (1982). Least Squares Quantization in PCM. _IEEE Trans. Inf. Theory_, 28(2), 129–137.
5. Comaniciu, D., & Meer, P. (2002). Mean Shift: A Robust Approach Toward Feature Space Analysis. _IEEE TPAMI_, 24(5), 603–619.
6. McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection. _arXiv:1802.03426_.
7. Rousseeuw, P. J. (1987). Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis. _J. Comput. Appl. Math._, 20, 53–65.
8. Davies, D. L., & Bouldin, D. W. (1979). A Cluster Separation Measure. _IEEE TPAMI_, PAMI-1(2), 224–227.
9. Reimers, N., & Gurevych, I. (2019). Sentence-BERT. _EMNLP-IJCNLP 2019_, 3982–3992.
10. Schubert, E. et al. (2017). DBSCAN Revisited, Revisited. _ACM TODS_, 42(3), 1–21.
11. Arthur, D., & Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding. _SODA_, 1027–1035.
12. Caliński, T., & Harabasz, J. (1974). A Dendrite Method for Cluster Analysis. _Commun. Stat._, 3(1), 1–27.
