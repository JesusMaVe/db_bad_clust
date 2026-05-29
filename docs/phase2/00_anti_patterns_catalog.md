# Catálogo de Anti-Patrones — Fase 1 (Data Generation)

## Propósito

Documentar todas las malas prácticas de base de datos implementadas en las 10 tablas generadas por `anti_patterns.py`. Este catálogo sirve como ground truth para el pipeline de ML: los embeddings y clusters deben reflejar estas categorías de anti-patrones.

## Ubicación

`scripts/anti_patterns.py`

## Quick Start

```bash
docker compose up -d && sleep 30
cd scripts && python orchestrator.py
```

## Arquitectura

```
anti_patterns.py
├── AntiPatternTable     → definición de esquema (columnas, tipos, flags)
├── AntiPatternData      → datos INSERT con dirty data
├── generate_poorly_designed_tables() → genera 10 tablas
└── generate_bad_data_for_table()     → genera datos corruptos por tabla
```

## Las 13 Tablas

### 1. EMPLEADOS — Tipos incorrectos + columnas fantasma

| Columna | Tipo declarado | Problema |
|---|---|---|
| ID | NUMBER | 30% NULL |
| NOMBRE_EMPLEADO | VARCHAR2(100) | HTML, espacios, tabs,®™ |
| FECHA_NACIMIENTO | VARCHAR2(20) | Fecha almacenada como texto |
| SALARIO | VARCHAR2(50) | Número como texto ($, euros, "Sin definir") |
| DEPARTAMENTO | NUMBER | Texto almacenado como número |
| EMAIL | DATE | Email almacenado como fecha |
| ACTIVO | VARCHAR2(10) | 6 convenciones booleanas distintas |
| COL_A_BORRAR_1 | VARCHAR2(4000) | Ghost column — siempre NULL |
| DUPLICADO_NOMBRE | VARCHAR2(100) | Redundante con NOMBRE_EMPLEADO |
| FECHA_ALTA | VARCHAR2(20) | Datos autocontradictorios (activo=No pero fecha futura) |

### 2. TBL_DATOS — Nombres crípticos

| Columna | Problema |
|---|---|
| C1–C7 | Sin significado semántico, indocumentadas |
| COL_EXTRA, TEMP_DATA | Nombres vagos, CLOB para texto trivial |

### 3. ORDENES_COMPRA — Tipos incorrectos + datos extremos

| Columna | Problema |
|---|---|
| FECHA_ORDEN VARCHAR2(50) | 9 formatos de fecha + fechas imposibles |
| CANTIDAD VARCHAR2(10) | Números como "varios", "N/A" |
| PRECIO VARCHAR2(20) | $, €, "Gratis" mezclados |
| ESTADO NUMBER | Texto y número mezclados |
| DIRECCION_ENVIO, TELF_CONTACTO | VARCHAR2(4000) para campos pequeños |

### 4. CLIENTES_DIRECCIONES — Desnormalización total

| Columna | Problema |
|---|---|
| DIRECCION_COMPLETA | Calle + número + piso todo en un campo |
| TELEFONOS | Múltiples teléfonos en 10 formatos distintos |
| EMAILS | Múltiples emails separados por coma |
| PROVINCIA_CIUDAD | Provincia y ciudad juntos |
| CODIGO_POSTAL_PAIS | CP y país juntos |
| FAX | Tecnología obsoleta, siempre NULL |

### 5. PRODUCTOS — Columnas multivalor

| Columna | Problema |
|---|---|
| PRECIOS | "19.99, 24.99, 29.99" en un campo |
| CATEGORIAS | Múltiples categorías separadas por coma |
| PROVEEDORES_ID | Lista de IDs como texto |
| TALLAS_DISPONIBLES | "S,M,L,XL" en un campo |
| PROVEEDOR_NOMBRE | Redundante con PROVEEDORES_ID |

### 6. TODO_EN_UNO — Tabla polimórfica gigante

Anti-patrones: **polimorfismo**, **tabla gigante**, **VARCHAR2(4000) en todo**, **4 booleanos con 4 convenciones**

| Columna | Problema |
|---|---|
| TIPO_REGISTRO | Define si la fila es CLIENTE, PRODUCTO, EMPLEADO, ORDEN o PROVEEDOR |
| NOMBRE_O_DESCRIPCION | Según TIPO_REGISTRO, es nombre o descripción |
| CANT_O_PRECIO | Según TIPO_REGISTRO, es cantidad o precio |
| FECHA_O_DIRECCION | Según TIPO_REGISTRO, es fecha o dirección |
| ESTADO_O_ACTIVO | VARCHAR2(4000) para un flag |
| FLAG_S_N / Y_N / 1_0 / T_F | 4 columnas booleanas con 4 convenciones distintas |

### 7. CATEGORIAS — Auto-referencia cíclica

Anti-patrones: **self-referencing FK**, **datos cíclicos**, **nivel como texto**

| Columna | Problema |
|---|---|
| ID | Sin PK declarada |
| CAT_PADRE_ID | Crea ciclos (A→B→C→A), apunta a IDs inexistentes (99) |
| NIVEL VARCHAR2(10) | Mezcla "0", "1", "Dos", "tres", NULL |
| ACTIVO CHAR(1) | Y/N/S/1/0 mezclados |

### 8. TRANSACCIONES — Datos imposibles

Anti-patrones: **IDs negativos**, **fechas imposibles**, **precisión absurda**, **TIPO críptico**

| Columna | Problema |
|---|---|
| ID | Negativo (-5, -10) o NULL |
| MONTO VARCHAR2(50) | Con 20 decimales, con símbolos, "Gratis" |
| FECHA VARCHAR2(30) | "9999-12-31", "0001-01-01", "ayer", "mañana", "Fecha inválida" |
| TIPO CHAR(1) | ~15 valores (A-J, 1, 2, 9, X, Z) sin documentación |
| MONEDA | USD, EUR, MXN, dólares, $, $USD, NULL mezclados |
| CONFIRMADO CHAR(1) | Y/N/S/1/0 mezclados |

### 9. EMPLEADOS_HISTORIAL — Datos autocontradictorios

Anti-patrones: **sin FK real**, **salarios contradictorios**, **tipos de cambio engañosos**

| Columna | Problema |
|---|---|
| EMPLEADO_ID | Apunta a IDs que no existen en EMPLEADOS (sin FK) |
| SALARIO_ANTERIOR / NUEVO | SALARIO_NUEVO < SALARIO_ANTERIOR pero TIPO_CAMBIO="aumento" |
| FECHA_CAMBIO | Fecha imposible o NULL |
| TIPO_CAMBIO | "DESCONOCIDO", NULL, "", "  " |

### 10. CONFIGURACION — EAV anti-patrón

Anti-patrones: **EAV sin validación**, **VALOR multiformato**, **CLAVE inconsistente**

| Columna | Problema |
|---|---|
| CLAVE | Mezcla UPPER_SNAKE, camelCase, lower_snake, NULL, "  " |
| VALOR VARCHAR2(4000) | Almacena JSON, XML, base64, números, secretos — todo como texto |
| TIPO_DATO | "NUMERO", "EMAIL", "JSON", "XML", "BASE64", "DESCONOCIDO", NULL |

### 11. DEPARTAMENTOS — Tabla con FK rota

Anti-patrones: **FK creada pero sin PK en destino** (EMPLEADOS.DEPARTAMENTO → DEPARTAMENTOS.ID)

| Columna | Problema |
|---|---|
| ID | Sin PK |
| PRESUPUESTO VARCHAR2(20) | Número como texto, NULL, "Sin presupuesto" |

### 12. PROVEEDORES — Tabla fantasma para relación rota

Anti-patrones: **PRODUCTOS.PROVEEDORES_ID debería referenciar esta tabla pero no hay FK** — referencia fantasma

| Columna | Problema |
|---|---|
| ID | Sin PK |
| CONTACTO | HTML, NULL, multi-byte |
| TELEFONO | 10 formatos distintos |

### 13. AUDITORIA_LOG — FK apuntando a tabla sin PK

Anti-patrones: **FK DISABLE a TODO_EN_UNO que no tiene PK** — constraint que no puede funcionar

| Columna | Problema |
|---|---|
| REGISTRO_ID | FK DISABLE a TODO_EN_UNO(ID) — imposible de activar |
| TABLA_AFECTADA | "EMPLEADOS", "PRODUCTOS", NULL, "" |
| FECHA | Timezones inconsistentes |
| DETALLE | Multi-byte, XSS, NULL |

## Tipos de Anti-Patrones por Categoría

### Esquema (14 tipos)
| Anti-patrón | Tablas |
|---|---|
| Sin Primary Key | TODAS (13/13) |
| Tipos incorrectos | DEPARTAMENTOS, EMPLEADOS, ORDENES_COMPRA, TODO_EN_UNO, CATEGORIAS, TRANSACCIONES, EMPLEADOS_HISTORIAL |
| Nombres crípticos | TBL_DATOS |
| Desnormalización | CLIENTES_DIRECCIONES |
| Columnas multivalor | PRODUCTOS |
| Polimorfismo | TODO_EN_UNO |
| Tabla gigante (>15 cols) | TODO_EN_UNO |
| Self-referencing | CATEGORIAS |
| EAV anti-patrón | CONFIGURACION |
| Columnas fantasma | EMPLEADOS, ORDENES_COMPRA, CLIENTES_DIRECCIONES, PRODUCTOS |
| FK rota (sin PK destino) | EMPLEADOS → DEPARTAMENTOS |
| FK DISABLE (sin PK destino) | AUDITORIA_LOG → TODO_EN_UNO |
| CHECK DISABLE contradictorio | EMPLEADOS (SALARIO>0 AND SALARIO<0), TRANSACCIONES (MONTO>=0), PRODUCTOS (PRECIOS NOT NULL), CATEGORIAS (ACTIVO IN ('S','N')) |
| Índices redundantes | EMPLEADOS (5 índices: 3 en ID, 2 en NOMBRE_EMPLEADO) |
| Relación fantasma (sin FK) | PRODUCTOS.PROVEEDORES_ID → PROVEEDORES.ID |

### Datos (~15 tipos)
| Anti-patrón | Descripción |
|---|---|
| NULLs en required | IDs, nombres, fechas NULL |
| HTML/XML en texto | `<b>`, `<p>`, `<script>alert('xss')</script>` |
| Fechas inconsistentes | 9 formatos distintos mezclados |
| Timezones inconsistentes | ISO8601 con Z, +HH:MM, EST, sin tz, solo offset |
| Fechas imposibles | 9999-12-31, 0001-01-01, "ayer", "mañana" |
| Espacios | Leading, trailing, tabs, newlines |
| Caracteres especiales | ®, ™, Unicode |
| Duplicados | Filas exactas duplicadas |
| Nulos semánticos | 0, -1, 9999, 'N/A', 'Sin definir', '' mezclados |
| Autocontradicción | activo=No pero fecha_futura; "aumento" con salario menor |
| IDs negativos | -5, -10 en TRANSACCIONES |
| Precisión absurda | 20 decimales en montos |
| Tipos mezclados | $, EUR, "Gratis" en misma columna numérica |
| Booleanos inconsistentes | S/N, Y/N, 1/0, T/F, true/false mezclados |
| SQL injection | `<script>` tags en datos |
| Formato teléfonos | 10+ formatos distintos (+34, (555), ext, etc.) |
| Multi-byte | Latin1 (ñ,ü) + CJK (データ) + Emojis (🚀) mezclados |

## Lo que se logró

1. **De 5 a 13 tablas** cubriendo 14 anti-patrones de esquema y ~17 de datos
2. **FK rotas**: 3 casos distintos (FK sin PK destino, FK DISABLE, relación fantasma)
3. **CHECK constraints contradictorias**: 4 constraints DISABLE que nunca podrían funcionar
4. **Índices redundantes**: 5 índices duplicados en EMPLEADOS (3 en ID, 2 en NOMBRE)
5. **Timezones inconsistentes**: 8 formatos de zona horaria mezclados en fechas
6. **Multi-byte**: Latin1 + CJK + Emojis mezclados en nombres y descripciones
7. **~220 filas totales** con densidad alta de corrupción
8. **Listo para ML**: el pipeline Phase 2 puede consumir estas 13 tablas

## Próximos pasos

### Módulos Phase 2 pendientes (6-10)
- [ ] **Módulo 6 — DimensionalityReducer**: Reducción de dimensionalidad (PCA/t-SNE/UMAP) a los vectores compuestos
- [ ] **Módulo 7 — ClusterEngine**: Algoritmos de clustering sobre los vectores reducidos
- [ ] **Módulo 8 — Evaluator**: Métricas de evaluación de calidad del clustering
- [ ] **Módulo 9 — Recommender**: Recomendación de anti-patrones basada en clusters
- [ ] **Módulo 10 — Visualizer**: Visualización de resultados (matrícula de correlación, scatter plots, etc.)
