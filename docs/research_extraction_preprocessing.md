# Investigación: extracción de metadatos de esquema Oracle y preprocesamiento de texto para embeddings

**Fecha:** 2026-08-18
**Método:** revisión de fuentes primarias (documentación oficial de Oracle, Hugging Face, sentence-transformers, papers en arXiv, documentación de DataHub/SchemaSpy).
**Alcance:** (1) best practices de extracción de metadata del data dictionary de Oracle 23c; (2) best practices de preprocesamiento de texto de identificadores para embeddings semánticos.

---

## Tema 1: Extracción de metadatos de esquema (Oracle data dictionary)

### 1.1 `USER_*` vs `ALL_*` vs `DBA_*`: cuándo usar cada una

- Oracle agrupa las vistas del data dictionary en conjuntos de tres con prefijos `DBA_`, `ALL_` y `USER_`. `DBA_` muestra todos los objetos de la base de datos (solo administradores); `ALL_` muestra los objetos a los que el usuario tiene acceso por grants públicos o explícitos (incluye los propios); `USER_` muestra solo los objetos del usuario actual y es un subconjunto de `ALL_` (la columna `OWNER` está implícita). Fuente: [Oracle Database Concepts 19c, Table 7-1](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/data-dictionary-and-dynamic-performance-views.html)
- Las vistas `ALL_*` obedecen al conjunto de roles habilitados en la sesión: el resultado puede cambiar según los roles activos (el manual lo demuestra con `SET ROLE NONE` cambiando el conteo de `ALL_OBJECTS`). Implicación práctica: el resultado de una extracción con `ALL_*` no es estable entre sesiones/roles. Fuente: [Oracle Concepts, "Views with the Prefix ALL_"](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/data-dictionary-and-dynamic-performance-views.html)
- Las páginas de referencia de las vistas `USER_*` delegan en las `ALL_*` ("sus columnas son las mismas que en ALL_..."), es decir, son intercambiables salvo por el filtro de owner. Ejemplos: [USER_CONSTRAINTS](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/USER_CONSTRAINTS.html), [USER_CONS_COLUMNS](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/USER_CONS_COLUMNS.html), [USER_TAB_COMMENTS](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_TAB_COMMENTS.html), [USER_COL_COMMENTS](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_COL_COMMENTS.html)
- **Regla práctica soportada por las fuentes:** si el pipeline se conecta con el usuario dueño del schema generado (nuestro caso: la BD "oracle" de malas prácticas se crea con el usuario de conexión), `USER_*` es la elección correcta y más simple. Para catalogar schemas de otros owners se necesita `ALL_*` (con grants) o `DBA_*` (con `SELECT_CATALOG_ROLE` o `SELECT ANY DICTIONARY`). DataHub, el data catalog de referencia, implementa exactamente esta distinción con su opción `data_dictionary_mode: ALL | DBA`. Fuente: [DataHub docs — Oracle](https://datahubproject.io/docs/generated/ingestion/sources/oracle/)
- En arquitecturas multitenant (CDB/PDB) hay que conectarse directamente al PDB por `service_name` y aplicar los grants dentro de cada PDB. Fuente: [DataHub docs — Oracle, sección Multitenant](https://datahubproject.io/docs/generated/ingestion/sources/oracle/)

### 1.2 Constraints, índices y comentarios: qué cubre cada vista y qué falta

**Constraints — `USER_CONSTRAINTS` + `USER_CONS_COLUMNS` es el patrón correcto, pero incompleto:**

- `ALL_CONS_COLUMNS` da `COLUMN_NAME` y `POSITION` (posición de la columna en la definición de la constraint) — `POSITION` es imprescindible para reconstruir PK/UK/FK compuestos en orden. Fuente: [ALL_CONS_COLUMNS](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_CONS_COLUMNS.html)
- `ALL_CONSTRAINTS` incluye metadatos que un extractor ingenuo suele omitir:
  - `CONSTRAINT_TYPE` distingue `C` (check), `P` (PK), `U` (unique), `R` (FK), `V`/`O` (vistas), `H`, `F`, `S`. Fuente: [ALL_CONSTRAINTS](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_CONSTRAINTS.html)
  - El texto de los CHECK está en `SEARCH_CONDITION`, de tipo `LONG` (problemático de manipular); existe `SEARCH_CONDITION_VC VARCHAR2(4000)` como alternativa manejable, aunque puede truncar. Fuente: [ALL_CONSTRAINTS](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_CONSTRAINTS.html)
  - Para FKs: `DELETE_RULE` (`CASCADE`/`SET NULL`/`NO ACTION`), `R_OWNER`/`R_CONSTRAINT_NAME` (la constraint referenciada — así se resuelve a qué tabla/columna apunta el FK), `STATUS`, `DEFERRABLE`, `VALIDATED`, `GENERATED` (nombre generado por el sistema). Fuente: [ALL_CONSTRAINTS](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_CONSTRAINTS.html)
  - `INDEX_NAME`/`INDEX_OWNER` (solo para PK/unique): el índice que respalda la constraint. Fuente: [ALL_CONSTRAINTS](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_CONSTRAINTS.html)

**Índices — `USER_INDEXES` + `USER_IND_COLUMNS` se queda corto:**

- `ALL_INDEXES` aporta señales útiles para detectar anti-patrones: `UNIQUENESS`, `INDEX_TYPE` (incluye `FUNCTION-BASED *`, `BITMAP`, `DOMAIN`), `FUNCIDX_STATUS` (estado de índices funcionales), `GENERATED` (índice creado por una constraint), `VISIBILITY`, `PARTITIONED`. Fuente: [ALL_INDEXES](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_INDEXES.html)
- `ALL_IND_COLUMNS` da `COLUMN_POSITION`, `DESCEND` (orden ASC/DESC), `COLUMN_LENGTH`. Fuente: [ALL_IND_COLUMNS](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_IND_COLUMNS.html)
- **Falta importante:** para índices function-based, `*_IND_COLUMNS` no muestra la expresión; hay que consultar `ALL_IND_EXPRESSIONS` (`COLUMN_EXPRESSION`, otra columna `LONG`). Fuente: [ALL_IND_EXPRESSIONS](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_IND_EXPRESSIONS.html)

**Comentarios:**

- `ALL_TAB_COMMENTS` (comentarios de tablas/vistas, `COMMENTS VARCHAR2(4000)`). Fuente: [ALL_TAB_COMMENTS](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_TAB_COMMENTS.html)
- `ALL_COL_COMMENTS` (comentarios por columna, `COMMENTS VARCHAR2(4000)`). Fuente: [ALL_COL_COMMENTS](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_COL_COMMENTS.html)
- Nota de rendimiento: según Oracle Concepts, las columnas `COMMENTS` no se cachean en el data dictionary cache (solo en el database buffer cache). Fuente: [Oracle Concepts, "Data Dictionary Cache"](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/data-dictionary-and-dynamic-performance-views.html)

**Columnas — `USER_TAB_COLUMNS` vs `USER_TAB_COLS`:**

- `ALL_TAB_COLUMNS` filtra las columnas ocultas generadas por el sistema; `ALL_TAB_COLS` no las filtra y añade `HIDDEN_COLUMN`, `VIRTUAL_COLUMN`, `USER_GENERATED`, `QUALIFIED_COL_NAME`. Las columnas virtuales solo se identifican con el flag `VIRTUAL_COLUMN` de `*_TAB_COLS`. Fuentes: [ALL_TAB_COLUMNS](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_TAB_COLUMNS.html), [ALL_TAB_COLS](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_TAB_COLS.html)
- Ambas vistas incluyen señales que un extractor ingenuo omite: `NULLABLE`, `COLUMN_ID` (orden de creación), `DATA_PRECISION`/`DATA_SCALE`, `CHAR_LENGTH` y `CHAR_USED` (semántica BYTE/CHAR), `DATA_DEFAULT` (valor por defecto, **tipo `LONG`**), `DEFAULT_ON_NULL`, `IDENTITY_COLUMN`, `COLLATION`. Fuente: [ALL_TAB_COLUMNS](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_TAB_COLUMNS.html)
- Identity columns: además del flag `IDENTITY_COLUMN`, `ALL_TAB_IDENTITY_COLS` da `GENERATION_TYPE` (`ALWAYS`/`BY DEFAULT`), la `SEQUENCE_NAME` interna y `IDENTITY_OPTIONS`. Fuente: [ALL_TAB_IDENTITY_COLS](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_TAB_IDENTITY_COLS.html)
- Particiones: `ALL_TAB_PARTITIONS` da `PARTITION_NAME`, `PARTITION_POSITION`, `HIGH_VALUE` (LONG), `SUBPARTITION_COUNT`, etc. Fuente: [ALL_TAB_PARTITIONS](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_TAB_PARTITIONS.html)
- Secuencias: `ALL_SEQUENCES` (min/max, `INCREMENT_BY`, `CYCLE_FLAG`, `CACHE_SIZE`, `SCALE_FLAG`, `SESSION_FLAG`...). Fuente: [ALL_SEQUENCES](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_SEQUENCES.html)
- Triggers: `ALL_TRIGGERS` (tipo, evento disparador, tabla base, `STATUS`, `TRIGGER_BODY` — otra `LONG`). Fuente: [ALL_TRIGGERS](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_TRIGGERS.html)
- Para el DDL completo de cualquier objeto, Oracle proporciona `DBMS_METADATA` (salida XML o SQL DDL); es lo que usan las herramientas de ingeniería inversa. Fuente: [Oracle Concepts, "Database Object Metadata"](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/data-dictionary-and-dynamic-performance-views.html)

### 1.3 Rendimiento: N+1 por tabla vs una query bulk + agrupación

- El data dictionary se consulta con SQL normal y las vistas son ya agregaciones con joins sobre tablas base normalizadas; además gran parte de la información vive en el data dictionary cache (la de parsing). Cada consulta al dictionary es una sentencia SQL completa con sus round-trips. Fuentes: [Oracle Concepts — Contents of the Data Dictionary](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/data-dictionary-and-dynamic-performance-views.html), [Data Dictionary Cache](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/data-dictionary-and-dynamic-performance-views.html)
- No existe una "regla oficial" N+1-vs-bulk en la documentación de Oracle; el patrón observable en herramientas profesionales es **una consulta por vista del dictionary y agrupación en el cliente**, no una query por tabla. DataHub documenta su conjunto fijo de vistas y hace una pasada por cada una (lista exacta en su página de Oracle). Fuente: [DataHub docs — Oracle](https://datahubproject.io/docs/generated/ingestion/sources/oracle/)
- **Conclusión:** el patrón actual del proyecto (queries bulk por tipo de vista + agrupación) es el correcto y es el que sigue la industria. El N+1 por tabla es aceptable con 23 tablas pero no escala y no aporta nada. Conviene saber que columnas `LONG` (`DATA_DEFAULT`, `SEARCH_CONDITION`, `COLUMN_EXPRESSION`, `TRIGGER_BODY`) deben leerse tal cual en el SELECT y agruparse/filtrarse en Python, no en SQL.

### 1.4 Qué extraen las herramientas profesionales que un extractor ingenuo omite

| Herramienta | Metadata que extrae | Fuente |
|---|---|---|
| **DataHub (oracle)** | Tablas y vistas con columnas, constraints y comentarios; usa `ALL_TAB_COLS` (no `ALL_TAB_COLUMNS`), `ALL_TAB_COMMENTS`, `ALL_COL_COMMENTS`, `ALL_TAB_IDENTITY_COLS`, `ALL_CONSTRAINTS`, `ALL_CONS_COLUMNS`; además procedimientos almacenados con código fuente (`ALL_SOURCE`), argumentos (`ALL_ARGUMENTS`), dependencias (`ALL_DEPENDENCIES`), materialized views (`ALL_MVIEWS`), usuarios/schemas (`ALL_USERS`) y estadísticas de uso (`V$SQL`) | [DataHub docs — Oracle](https://datahubproject.io/docs/generated/ingestion/sources/oracle/) |
| **SchemaSpy** | Analizador de metadata vía JDBC: diagramas ER, estadísticas de estructura y, sobre todo, **detección de anomalías**: "missing indexes, implied relationships, and orphan tables" (tablas sin PK, relaciones implícitas sin FK, columnas sin índice) | [SchemaSpy GitHub README](https://github.com/schemaspy/schemaspy) |
| **Oracle SQL Developer / Data Modeler** | Solución completa de data modeling: ERDs, DDL scripting, reports y DIFFs (ingeniería inversa del data dictionary); el DDL completo sale de `DBMS_METADATA` | [Página oficial SQL Developer](https://www.oracle.com/database/sqldeveloper/), [Oracle Concepts — DBMS_METADATA](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/data-dictionary-and-dynamic-performance-views.html) |
| **Apache Atlas** | Framework de gobernanza y metadata para el ecosistema Hadoop (integración por hooks/bridges, no trae extractor nativo del dictionary de Oracle) | [atlas.apache.org](https://atlas.apache.org/index.html) |

**Síntesis — lo que las herramientas profesionales extraen y un extractor ingenuo suele perder:** comentarios de tabla y columna, valor por defecto (`DATA_DEFAULT`), `DEFAULT_ON_NULL`, columnas virtuales y ocultas, columnas identity, texto de CHECK constraints, `DELETE_RULE` y estado de FKs, índices únicos/function-based (con sus expresiones), particiones, secuencias, triggers, materialized views, vistas y su definición, y objetos PL/SQL con dependencias.

---

## Tema 2: Preprocesamiento de texto para embeddings de identificadores

### 2.1 ¿El sub-word tokenization (WordPiece) ya divide camelCase/snake_case?

- Los modelos de la familia BERT usan **WordPiece**, que descompone palabras raras en sub-words de vocabulario compacto. Fuente: [HF Transformers — Summary of tokenization algorithms](https://huggingface.co/docs/transformers/tokenizer_summary)
- El pre-tokenizer de BERT (`BertPreTokenizer`) "divide en whitespace y puntuación; cada carácter de puntuación se trata como token separado". Fuente: [HF Tokenizers — Pre-tokenizers](https://huggingface.co/docs/tokenizers/api/pre-tokenizers)
- Consecuencias verificables en esa documentación:
  - **snake_case ya se divide solo**: el guion bajo es un carácter de puntuación, así que `user_id` llega al WordPiece como `user`, `_`, `id`. Pre-dividir snake_case es **redundante** (aunque inofensivo: evita tokens `_`).
  - **camelCase NO se divide**: no existe ninguna regla de cambio de mayúsculas en `BertPreTokenizer`; `userId` entra como una sola "palabra" y WordPiece la trocea de forma arbitraria (`user` + `##Id` o peor). Pre-dividir camelCase **sí aporta valor**.
- La normalización de BERT es `BertNormalizer(clean_text, handle_chinese_chars, strip_accents, lowercase)` según la configuración del tokenizer en el código fuente; no incluye normalización Unicode NFKC. Fuente: [tokenization_bert.py (transformers, GitHub)](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/bert/tokenization_bert.py)

### 2.2 [CLS] vs mean pooling vs sentence-transformers

- El paper de SBERT lo dice literalmente: *"The construction of BERT makes it unsuitable for semantic similarity search as well as for unsupervised tasks like clustering"*. Fuente: [arXiv:1908.10084 (abstract)](https://arxiv.org/abs/1908.10084)
- Sobre la práctica común de usar [CLS] o el promedio de la última capa de BERT sin fine-tuning: *"this common practice yields rather bad sentence embeddings, often worse than averaging GloVe embeddings"*. En su Tabla 1 (Spearman ×100, media de 7 tareas STS): **BERT CLS-vector = 29.19**, Avg. BERT embeddings = 54.81, Avg. GloVe = 61.32, **SBERT-NLI-base = 74.89**. Fuente: [SBERT paper, sección 1 y Tabla 1](https://arxiv.org/html/1908.10084v1)
- Conclusión del propio paper: *"average BERT embeddings / CLS-token output from BERT return sentence embeddings that are infeasible to be used with cosine-similarity or with Manhatten / Euclidean distance"*. Fuente: [SBERT paper, sección 5](https://arxiv.org/html/1908.10084v1)
- SBERT experimenta con tres pooling strategies (CLS, MEAN, MAX) y *"the default configuration is MEAN"*. Fuente: [SBERT paper, sección 3](https://arxiv.org/html/1908.10084v1)
- Matiz relevante: cuando encima del embedding se entrena un clasificador (SentEval, regresión logística), CLS/avg de BERT dan resultados decentes ([sección 5](https://arxiv.org/html/1908.10084v1)). Pero nuestro pipeline hace **clustering no supervisado con cosine/euclidean** — exactamente el escenario en el que el paper declara inviable el [CLS] de BERT crudo.
- Los modelos sentence-transformers de producción usan mean pooling (con attention mask) por defecto: `paraphrase-multilingual-MiniLM-L12-v2` tiene `pooling_mode_cls_token: False, pooling_mode_mean_tokens: True`; `distiluse-base-multilingual-cased-v1` añade una capa `Dense(768→512, tanh)` tras el mean pooling. Fuentes: [model card paraphrase-multilingual-MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), [model card distiluse-base-multilingual-cased-v1](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v1)

### 2.3 ¿Es `bert-base-multilingual-cased` la elección correcta?

- `bert-base-multilingual-cased` (mBERT) es un modelo MLM preentrenado sobre las 104 Wikipedias mayores, *"case sensitive: it makes a difference between english and English"* y *"mostly intended to be fine-tuned on a downstream task"*. No está entrenado para producir embeddings de frase comparables con cosine similarity. Fuentes: [model card bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased), [SBERT paper](https://arxiv.org/abs/1908.10084)
- **Alternativas multilingües de sentence-transformers (español + inglés cubiertos):**
  - `paraphrase-multilingual-MiniLM-L12-v2`: 384 dimensiones, entrenado con datos paralelos en 50+ idiomas, *"can be used for tasks like clustering or semantic search"*, `max_seq_length` 128, sin lowercasing. Los modelos multilingües de esta familia producen *"similar embeddings for the same texts in different languages"* (destilación multilingüe). Fuentes: [model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), [sbert.net — Pretrained Models, Multilingual](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
  - `distiluse-base-multilingual-cased-v1`: 512 dimensiones, destilación del multilingual Universal Sentence Encoder, soporta 15 idiomas (incluye español); la v2 cubre 50+ idiomas pero *"performs a bit weaker than the v1"*. Fuentes: [model card](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v1), [sbert.net — Multilingual Models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
  - `intfloat/multilingual-e5-base` (recomendación moderna): inicializado desde XLM-RoBERTa, ~100 idiomas (94 tags en el card), mean pooling + L2 normalization, entrenado contrastivamente con miles de millones de pares multilingües; su FAQ indica usar el prefijo `"query: "` incluso para clustering. Fuente: [model card multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)
- Para código/identificadores existe una línea específica (CodeBERT, modelo bimodal PL-NL para code search), pero está entrenado sobre CodeSearchNet con natural language en inglés; no resuelve el caso español/inglés mezclado de nombres de columnas. Fuente: [arXiv:2002.08155](https://arxiv.org/abs/2002.08155)

### 2.4 Expansión de abreviaturas e investigación sobre identificadores

- Dividir identificadores en palabras que preserven la semántica beneficia a las herramientas: *"splitting compound identifiers into sub-words that reflect the semantics can benefit software development tools"*. Pero cuidado: insertar identifier splitting ingenuamente en un pipeline puede empeorar el rendimiento; una estrategia híbrida (splitting + BPE) mejora los modelos. Fuente: [Can Identifier Splitting Improve Open-Vocabulary Language Model of Code? (arXiv:2201.01988)](https://arxiv.org/abs/2201.01988)
- Las palabras resultantes de dividir nombres de funciones llevan señal semántica suficiente para tareas de clasificación (las "palabras peligrosas" predicen funciones vulnerables). Fuente: [Featherweight Assisted Vulnerability Discovery (arXiv:2202.02679)](https://arxiv.org/abs/2202.02679)
- Sobre abreviaturas: *"Source code identifiers often contain abbreviations"*; la expansión de abreviaturas se ha investigado intensivamente y la mayoría de enfoques existentes son **heurísticas**; además, el contexto (el código circundante) es clave para desambiguar la expansión correcta. Fuentes: [Evaluating and Improving ChatGPT-Based Expansion of Abbreviations (arXiv:2410.23866)](https://arxiv.org/abs/2410.23866)
- Implicación para el pipeline: mantener la tabla manual de ~65 abreviaturas es consistente con el estado del arte (las heurísticas dominan el área), y contextualizar cada abreviatura con la tabla y el tipo de dato (lo que ya hacemos al anteponer el nombre de la tabla) reproduce el patrón de "usar contexto para desambiguar" que la literatura recomienda.

### 2.5 Lowercasing, truncation y normalización

- **No lowercasar con modelos cased.** mBERT-cased es explícitamente case sensitive ([model card](https://huggingface.co/bert-base-multilingual-cased)); los sentence-transformers multilingües citados llevan `do_lower_case: False` en su configuración ([paraphrase-multilingual-MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)). Lowercasar la entrada de un modelo cased destruye señal (p. ej. `ID` como acrónimo vs `Id`).
- **Truncation/padding explícitos.** La API de HF permite y recomienda fijar `truncation=True` y `max_length` al tokenizar; los modelos ST tienen `max_seq_length` 128 (paraphrase-MiniLM) y e5 512. Para textos cortos tipo `"auditoria log: identifier"` nunca se activa, pero fijarlo es gratis y elimina fallos silenciosos. Fuentes: [HF — Padding and truncation](https://huggingface.co/docs/transformers/main/en/pad_truncation), [model cards citados]
- **Mean pooling debe respetar la attention mask** (excluir padding del promedio); el patrón correcto está en el ejemplo del model card de paraphrase-multilingual-MiniLM. Fuente: [model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **NFKC**: la normalización Unicode no forma parte del normalizador BERT (ver 2.1); con identificadores del dictionary de Oracle (típicamente ASCII) no es crítico, pero es una normalización barata si aparecen caracteres fullwidth o acentos exóticos.

---

## Qué está mal o se puede mejorar en nuestro pipeline

Pipeline actual: queries bulk por tipo de vista (`user_tables`, `user_tab_columns`, `user_constraints`, `user_cons_columns`, `user_ind_columns`), split de camelCase/snake_case, ~65 abreviaturas expandidas manualmente, lowercasing, contextualización con nombre de tabla, `bert-base-multilingual-cased` con token [CLS].

1. **[CLS] de mBERT crudo es el punto más débil (crítico).** El paper de SBERT demuestra que el CLS de BERT sin fine-tuning produce los peores embeddings de frase (29.19 vs 61.32 de GloVe avg vs 74.89 de SBERT) y declara a BERT *"unsuitable... for unsupervised tasks like clustering"* — que es exactamente nuestro caso. Cambiar a un modelo sentence-transformers multilingüe con mean pooling (`paraphrase-multilingual-MiniLM-L12-v2`, o `multilingual-e5-base` con prefijo `"query: "` para más calidad) es el cambio de mayor impacto esperado. Fuentes: [SBERT](https://arxiv.org/html/1908.10084v1), [e5 card](https://huggingface.co/intfloat/multilingual-e5-base)
2. **Lowercasing con un modelo cased es contradictorio.** Tanto mBERT-cased como los ST multilingües esperan texto con caso (`do_lower_case: False`). Eliminar el lowercasing (o cambiar a un modelo uncased de forma coherente). Fuente: [mBERT card](https://huggingface.co/bert-base-multilingual-cased)
3. **Split de snake_case redundante, camelCase necesario.** `BertPreTokenizer` ya separa por puntuación (incluido `_`); no hay regla para camelCase. Mantener ambos splits no daña y da control del texto final, pero solo el de camelCase está justificado por las fuentes. Fuente: [HF Tokenizers](https://huggingface.co/docs/tokenizers/api/pre-tokenizers)
4. **Metadatos que no extraemos y valen la pena** (ordenado por valor para clasificar anti-patterns):
   - **Comentarios de tabla/columna** (`user_tab_comments`, `user_col_comments`): son texto natural en español/inglés — el mejor insumo para el embedding; además DataHub los extrae por defecto. Fuente: [DataHub](https://datahubproject.io/docs/generated/ingestion/sources/oracle/)
   - `DATA_DEFAULT` (tipo LONG), `DEFAULT_ON_NULL`, `IDENTITY_COLUMN` + `user_tab_identity_cols`, columnas virtuales/ocultas (`user_tab_cols` en vez de `user_tab_columns`; DataHub usa `ALL_TAB_COLS`). Fuentes: [ALL_TAB_COLUMNS](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_TAB_COLUMNS.html), [ALL_TAB_COLS](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_TAB_COLS.html)
   - Texto de CHECK constraints (`SEARCH_CONDITION_VC`), `DELETE_RULE`/`STATUS`/`DEFERRABLE` de FKs, índices function-based (`user_ind_expressions`), `UNIQUENESS`/`GENERATED` de índices, particiones, secuencias, triggers. Fuentes: [ALL_CONSTRAINTS](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_CONSTRAINTS.html), [ALL_INDEXES](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_INDEXES.html), [ALL_IND_EXPRESSIONS](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_IND_EXPRESSIONS.html)
   - Anomalías derivadas al estilo SchemaSpy: tablas sin PK, FKs implícitas (columnas con mismo nombre sin constraint), índices que duplican constraints — señales directas de anti-patterns. Fuente: [SchemaSpy README](https://github.com/schemaspy/schemaspy)
5. **Bulk queries: correcto, mantener.** Una query por vista + agrupación en Python es el patrón de DataHub; el N+1 por tabla no aporta nada y no escala. Las columnas LONG (`DATA_DEFAULT`, `SEARCH_CONDITION`, `COLUMN_EXPRESSION`, `TRIGGER_BODY`) hay que leerlas en el SELECT y procesarlas en el cliente. Fuentes: [DataHub](https://datahubproject.io/docs/generated/ingestion/sources/oracle/), [Oracle Concepts](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/data-dictionary-and-dynamic-performance-views.html)
6. **Expansión de abreviaturas: mantener, es práctica estándar.** La literatura sobre identificadores trabaja mayoritariamente con heurísticas y subraya el valor del contexto para desambiguar — nuestro contexto (tabla + tipo) va en la dirección correcta. Considerar enriquecer el texto a embeber con el comentario de columna cuando exista. Fuente: [arXiv:2410.23866](https://arxiv.org/abs/2410.23866)
7. **Truncation explícito** (`truncation=True, max_length=...`) al tokenizar: gratis y previene fallos silenciosos. Fuente: [HF pad/truncation](https://huggingface.co/docs/transformers/main/en/pad_truncation)

---

## Referencias

**Oracle (documentación oficial):**
- Data Dictionary and Dynamic Performance Views (Concepts): https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/data-dictionary-and-dynamic-performance-views.html
- ALL_CONSTRAINTS: https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_CONSTRAINTS.html
- ALL_CONS_COLUMNS: https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_CONS_COLUMNS.html
- USER_CONSTRAINTS: https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/USER_CONSTRAINTS.html
- USER_CONS_COLUMNS: https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/USER_CONS_COLUMNS.html
- ALL_TAB_COLUMNS: https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_TAB_COLUMNS.html
- ALL_TAB_COLS: https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_TAB_COLS.html
- ALL_INDEXES: https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_INDEXES.html
- ALL_IND_COLUMNS: https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_IND_COLUMNS.html
- ALL_IND_EXPRESSIONS: https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_IND_EXPRESSIONS.html
- ALL_TAB_COMMENTS: https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_TAB_COMMENTS.html
- ALL_COL_COMMENTS: https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_COL_COMMENTS.html
- ALL_TAB_PARTITIONS: https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_TAB_PARTITIONS.html
- ALL_SEQUENCES: https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_SEQUENCES.html
- ALL_TRIGGERS: https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_TRIGGERS.html
- ALL_TAB_IDENTITY_COLS: https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/ALL_TAB_IDENTITY_COLS.html

**Herramientas de análisis de esquemas:**
- DataHub — Oracle ingestion: https://datahubproject.io/docs/generated/ingestion/sources/oracle/
- SchemaSpy (GitHub README): https://github.com/schemaspy/schemaspy
- Oracle SQL Developer (incluye Data Modeler): https://www.oracle.com/database/sqldeveloper/
- Apache Atlas: https://atlas.apache.org/index.html

**Papers:**
- BERT: Pre-training of Deep Bidirectional Transformers (arXiv:1810.04805): https://arxiv.org/abs/1810.04805
- Sentence-BERT (arXiv:1908.10084): https://arxiv.org/abs/1908.10084 — texto completo: https://arxiv.org/html/1908.10084v1
- CodeBERT (arXiv:2002.08155): https://arxiv.org/abs/2002.08155
- Identifier splitting + BPE (arXiv:2201.01988): https://arxiv.org/abs/2201.01988
- Featherweight Assisted Vulnerability Discovery (arXiv:2202.02679): https://arxiv.org/abs/2202.02679
- ChatGPT-Based Expansion of Abbreviations (arXiv:2410.23866): https://arxiv.org/abs/2410.23866

**Hugging Face / sentence-transformers:**
- bert-base-multilingual-cased: https://huggingface.co/bert-base-multilingual-cased
- paraphrase-multilingual-MiniLM-L12-v2: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- distiluse-base-multilingual-cased-v1: https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v1
- intfloat/multilingual-e5-base: https://huggingface.co/intfloat/multilingual-e5-base
- sbert.net — Pretrained Models (multilingües): https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
- Transformers — Summary of tokenization algorithms: https://huggingface.co/docs/transformers/tokenizer_summary
- Transformers — Padding and truncation: https://huggingface.co/docs/transformers/main/en/pad_truncation
- Tokenizers — Pre-tokenizers (BertPreTokenizer): https://huggingface.co/docs/tokenizers/api/pre-tokenizers
- tokenization_bert.py (código fuente transformers): https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/tokenization_bert.py
