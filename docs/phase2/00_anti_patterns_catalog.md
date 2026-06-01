# Catalogo de Anti-Patrones  Fase 1 (Data Generation)

## Proposito

Documentar todas las malas practicas de base de datos implementadas en las 23 tablas generadas por `anti_patterns.py`. Este catalogo sirve como ground truth para el pipeline de ML: los embeddings y clusters deben reflejar estas categorias de anti-patrones.

**Fase 1 completada.**

## Ubicacion

`scripts/anti_patterns.py`

## Quick Start

```bash
docker compose up -d && sleep 30
cd scripts && python orchestrator.py
```

## Arquitectura

```
anti_patterns.py
|-- AntiPatternTable     -> definicion de esquema (columnas, tipos, flags)
|-- AntiPatternData      -> datos INSERT con dirty data
|-- generate_poorly_designed_tables() -> genera 10 tablas
\-- generate_bad_data_for_table()     -> genera datos corruptos por tabla
```

## Las 23 Tablas

### 1. EMPLEADOS  Tipos incorrectos + columnas fantasma

| Columna | Tipo declarado | Problema |
|---|---|---|
| ID | NUMBER | 30% NULL |
| NOMBRE_EMPLEADO | VARCHAR2(100) | HTML, espacios, tabs, |
| FECHA_NACIMIENTO | VARCHAR2(20) | Fecha almacenada como texto |
| SALARIO | VARCHAR2(50) | Numero como texto ($, euros, "Sin definir") |
| DEPARTAMENTO | NUMBER | Texto almacenado como numero |
| EMAIL | DATE | Email almacenado como fecha |
| ACTIVO | VARCHAR2(10) | 6 convenciones booleanas distintas |
| COL_A_BORRAR_1 | VARCHAR2(4000) | Ghost column  siempre NULL |
| DUPLICADO_NOMBRE | VARCHAR2(100) | Redundante con NOMBRE_EMPLEADO |
| FECHA_ALTA | VARCHAR2(20) | Datos autocontradictorios (activo=No pero fecha futura) |

### 2. TBL_DATOS  Nombres cripticos

| Columna | Problema |
|---|---|
| C1C7 | Sin significado semantico, indocumentadas |
| COL_EXTRA, TEMP_DATA | Nombres vagos, CLOB para texto trivial |

### 3. ORDENES_COMPRA  Tipos incorrectos + datos extremos

| Columna | Problema |
|---|---|
| FECHA_ORDEN VARCHAR2(50) | 9 formatos de fecha + fechas imposibles |
| CANTIDAD VARCHAR2(10) | Numeros como "varios", "N/A" |
| PRECIO VARCHAR2(20) | $, , "Gratis" mezclados |
| ESTADO NUMBER | Texto y numero mezclados |
| DIRECCION_ENVIO, TELF_CONTACTO | VARCHAR2(4000) para campos pequenyos |

### 4. CLIENTES_DIRECCIONES  Desnormalizacion total

| Columna | Problema |
|---|---|
| DIRECCION_COMPLETA | Calle + numero + piso todo en un campo |
| TELEFONOS | Multiples telefonos en 10 formatos distintos |
| EMAILS | Multiples emails separados por coma |
| PROVINCIA_CIUDAD | Provincia y ciudad juntos |
| CODIGO_POSTAL_PAIS | CP y pais juntos |
| FAX | Tecnologia obsoleta, siempre NULL |

### 5. PRODUCTOS  Columnas multivalor

| Columna | Problema |
|---|---|
| PRECIOS | "19.99, 24.99, 29.99" en un campo |
| CATEGORIAS | Multiples categorias separadas por coma |
| PROVEEDORES_ID | Lista de IDs como texto |
| TALLAS_DISPONIBLES | "S,M,L,XL" en un campo |
| PROVEEDOR_NOMBRE | Redundante con PROVEEDORES_ID |

### 6. TODO_EN_UNO  Tabla polimorfica gigante

Anti-patrones: **polimorfismo**, **tabla gigante**, **VARCHAR2(4000) en todo**, **4 booleanos con 4 convenciones**

| Columna | Problema |
|---|---|
| TIPO_REGISTRO | Define si la fila es CLIENTE, PRODUCTO, EMPLEADO, ORDEN o PROVEEDOR |
| NOMBRE_O_DESCRIPCION | Segun TIPO_REGISTRO, es nombre o descripcion |
| CANT_O_PRECIO | Segun TIPO_REGISTRO, es cantidad o precio |
| FECHA_O_DIRECCION | Segun TIPO_REGISTRO, es fecha o direccion |
| ESTADO_O_ACTIVO | VARCHAR2(4000) para un flag |
| FLAG_S_N / Y_N / 1_0 / T_F | 4 columnas booleanas con 4 convenciones distintas |

### 7. CATEGORIAS  Auto-referencia ciclica

Anti-patrones: **self-referencing FK**, **datos ciclicos**, **nivel como texto**

| Columna | Problema |
|---|---|
| ID | Sin PK declarada |
| CAT_PADRE_ID | Crea ciclos (A->B->C->A), apunta a IDs inexistentes (99) |
| NIVEL VARCHAR2(10) | Mezcla "0", "1", "Dos", "tres", NULL |
| ACTIVO CHAR(1) | Y/N/S/1/0 mezclados |

### 8. TRANSACCIONES  Datos imposibles

Anti-patrones: **IDs negativos**, **fechas imposibles**, **precision absurda**, **TIPO criptico**

| Columna | Problema |
|---|---|
| ID | Negativo (-5, -10) o NULL |
| MONTO VARCHAR2(50) | Con 20 decimales, con simbolos, "Gratis" |
| FECHA VARCHAR2(30) | "9999-12-31", "0001-01-01", "ayer", "manyana", "Fecha invalida" |
| TIPO CHAR(1) | ~15 valores (A-J, 1, 2, 9, X, Z) sin documentacion |
| MONEDA | USD, EUR, MXN, dolares, $, $USD, NULL mezclados |
| CONFIRMADO CHAR(1) | Y/N/S/1/0 mezclados |

### 9. EMPLEADOS_HISTORIAL  Datos autocontradictorios

Anti-patrones: **sin FK real**, **salarios contradictorios**, **tipos de cambio enganyosos**

| Columna | Problema |
|---|---|
| EMPLEADO_ID | Apunta a IDs que no existen en EMPLEADOS (sin FK) |
| SALARIO_ANTERIOR / NUEVO | SALARIO_NUEVO < SALARIO_ANTERIOR pero TIPO_CAMBIO="aumento" |
| FECHA_CAMBIO | Fecha imposible o NULL |
| TIPO_CAMBIO | "DESCONOCIDO", NULL, "", "  " |

### 10. CONFIGURACION  EAV anti-patron

Anti-patrones: **EAV sin validacion**, **VALOR multiformato**, **CLAVE inconsistente**

| Columna | Problema |
|---|---|
| CLAVE | Mezcla UPPER_SNAKE, camelCase, lower_snake, NULL, "  " |
| VALOR VARCHAR2(4000) | Almacena JSON, XML, base64, numeros, secretos  todo como texto |
| TIPO_DATO | "NUMERO", "EMAIL", "JSON", "XML", "BASE64", "DESCONOCIDO", NULL |

### 11. DEPARTAMENTOS  Tabla con FK rota

Anti-patrones: **FK creada pero sin PK en destino** (EMPLEADOS.DEPARTAMENTO -> DEPARTAMENTOS.ID)

| Columna | Problema |
|---|---|
| ID | Sin PK |
| PRESUPUESTO VARCHAR2(20) | Numero como texto, NULL, "Sin presupuesto" |

### 12. PROVEEDORES  Tabla fantasma para relacion rota

Anti-patrones: **PRODUCTOS.PROVEEDORES_ID deberia referenciar esta tabla pero no hay FK**  referencia fantasma

| Columna | Problema |
|---|---|
| ID | Sin PK |
| CONTACTO | HTML, NULL, multi-byte |
| TELEFONO | 10 formatos distintos |

### 13. AUDITORIA_LOG  FK apuntando a tabla sin PK

Anti-patrones: **FK DISABLE a TODO_EN_UNO que no tiene PK**  constraint que no puede funcionar

| Columna | Problema |
|---|---|
| REGISTRO_ID | FK DISABLE a TODO_EN_UNO(ID)  imposible de activar |
| TABLA_AFECTADA | "EMPLEADOS", "PRODUCTOS", NULL, "" |
| FECHA | Timezones inconsistentes |
| DETALLE | Multi-byte, XSS, NULL |

### 14. TABLA_BASE_DATOS  40 columnas con nombres monocolumna

Anti-patrones: **nombres a, b, c... aa, ab..., todas VARCHAR2(4000)**, **sin PK**, **sin sentido semantico**

| Columna | Problema |
|---|---|
| a, b, c, ..., z, aa0, ..., aa9, aba, ..., abd | Sin significado, todas VARCHAR2(4000), 40 columnas de puro ruido |

### 15. BACKUP_DATOS  50 columnas numeradas

Anti-patrones: **nombres COL_001..COL_050**, **todas VARCHAR2(4000)**, **ancho extremo**

| Columna | Problema |
|---|---|
| COL_001 .. COL_050 | 50 columnas sin significado, tamanyo fijo VARCHAR2(4000), todo NULL o basura |

### 16. DATOS_MAESTROS  Tipos deprecated

Anti-patrones: **LONG, LONG RAW, RAW(2000), NVARCHAR2(2000), CHAR(2000)**, **sobredimensionamiento masivo**

| Columna | Problema |
|---|---|
| DESCRIPCION LONG | LONG  tipo obsoleto, reemplazado por CLOB |
| DATOS_BINARIOS LONG RAW | LONG RAW  obsoleto, reemplazado por BLOB |
| CODIGO_HEX RAW(2000) | RAW(2000) para codigos pequenyos |
| FLAG_ACTIVO CHAR(2000) | 2000 caracteres para un flag binario |
| NOMBRE_LARGO NVARCHAR2(2000) | 2000 caracteres para nombres |

### 17. VENTAS  FK con type mismatch

Anti-patrones: **FK VARCHAR2 -> NUMBER**, **sin PK**, **precio como texto**

| Columna | Problema |
|---|---|
| PRODUCTO_FK VARCHAR2(20) | FK a PRODUCTOS(SKU) pero SKU es VARCHAR2  Oracle permite esto pero es fragil |
| PRECIO_TOTAL VARCHAR2(30) | Precio como texto con simbolos |
| FECHA_VENTA VARCHAR2(30) | Fecha como texto |

### 18. USUARIOS_WEB  Defaults absurdos

Anti-patrones: **DEFAULT 'NADA' en columna numerica**, **DEFAULT 0 en VARCHAR2**, **DEFAULT 12345 en VARCHAR2**

| Columna | Problema |
|---|---|
| INTENTOS_FALLIDOS VARCHAR2(10) DEFAULT 'NADA' | Default textual en lo que deberia ser numero |
| ACTIVO VARCHAR2(5) DEFAULT 0 | Default numerico en columna de texto |
| FECHA_EXPIRACION VARCHAR2(30) DEFAULT '9999-12-31' | Fecha imposible como default |
| ROL VARCHAR2(20) DEFAULT 12345 | Numero como default de texto |

### 19. INVENTARIO  Datos duplicados semanticamente

Anti-patrones: **STOCK, STOCK_2, STOCK_COPIA  mismo significado, distintos tipos**

| Columna | Problema |
|---|---|
| STOCK NUMBER | Almacena stock |
| STOCK_2 VARCHAR2(20) | Misma informacion, tipo distinto |
| STOCK_COPIA NUMBER | Misma informacion, nombre distinto |

### 20. REPORTES  Palabras reservadas como columnas

Anti-patrones: **columnas llamadas NULL, SELECT, FROM, WHERE**

| Columna | Problema |
|---|---|
| NULL VARCHAR2(100) | Palabra reservada como nombre |
| SELECT NUMBER | Palabra reservada |
| FROM, WHERE VARCHAR2(100) | Palabras reservadas |

### 21. METADATA  Overkill de tipos

Anti-patrones: **CLOB y BLOB para datos de 1 caracter**, **BLOB para flags booleanos**

| Columna | Problema |
|---|---|
| NOMBRE_CAMPO CLOB | CLOB para nombre corto |
| VALOR_CAMPO BLOB | BLOB para valores pequenyos |
| ES_ACTIVO CLOB | CLOB para booleano |
| FECHA_CREACION CLOB | CLOB para fecha |

### 22. TABLA_VACIA  Datos minimos

Anti-patrones: **tabla con 3 columnas**, **sin proposito**, **solo 1-2 filas utiles**

| Columna | Problema |
|---|---|
| ID | Sin PK, generalmente NULL |
| NOMBRE | NULL, HTML, espacios |
| ESTADO | Texto vago, NULL |

### 23. CACHE_TEMPORAL  Abuso de BLOB

Anti-patrones: **todo BLOB**, **fecha como VARCHAR2**, **sin PK**

| Columna | Problema |
|---|---|
| DATOS BLOB | Datos de cache |
| RESULTADO BLOB | Resultados de cache |
| TIMESTAMP VARCHAR2(30) | Fecha como texto |

## Tipos de Anti-Patrones por Categoria

### Esquema (20+ tipos)
| Anti-patron | Tablas |
|---|---|
| Sin Primary Key | TODAS (23/23) |
| Tipos incorrectos | DEPARTAMENTOS, EMPLEADOS, ORDENES_COMPRA, TODO_EN_UNO, CATEGORIAS, TRANSACCIONES, EMPLEADOS_HISTORIAL, VENTAS, USUARIOS_WEB, DATOS_MAESTROS, METADATA, CACHE_TEMPORAL |
| Nombres cripticos | TBL_DATOS, TABLA_BASE_DATOS, BACKUP_DATOS |
| Palabras reservadas | REPORTES (NULL, SELECT, FROM, WHERE) |
| Desnormalizacion | CLIENTES_DIRECCIONES |
| Columnas multivalor | PRODUCTOS |
| Polimorfismo | TODO_EN_UNO |
| Tabla gigante (>15 cols) | TODO_EN_UNO, TABLA_BASE_DATOS (40), BACKUP_DATOS (50) |
| Self-referencing | CATEGORIAS |
| EAV anti-patron | CONFIGURACION |
| Columnas fantasma | EMPLEADOS, ORDENES_COMPRA, CLIENTES_DIRECCIONES, PRODUCTOS |
| Datos duplicados semanticamente | INVENTARIO (STOCK x 3) |
| Defaults absurdos | USUARIOS_WEB |
| FK rota (sin PK destino) | EMPLEADOS -> DEPARTAMENTOS |
| FK DISABLE (sin PK destino) | AUDITORIA_LOG -> TODO_EN_UNO |
| FK type mismatch | VENTAS -> PRODUCTOS |
| CHECK DISABLE contradictorio | EMPLEADOS, TRANSACCIONES, PRODUCTOS, CATEGORIAS |
| Indices redundantes | EMPLEADOS (5 indices) |
| Relacion fantasma (sin FK) | PRODUCTOS.PROVEEDORES_ID -> PROVEEDORES.ID |
| Tipos deprecated | DATOS_MAESTROS (LONG, LONG RAW) |
| Overkill de tipos | METADATA (CLOB/BLOB para flags) |

### Datos (~15 tipos)
| Anti-patron | Descripcion |
|---|---|
| NULLs en required | IDs, nombres, fechas NULL |
| HTML/XML en texto | `<b>`, `<p>`, `<script>alert('xss')</script>` |
| Fechas inconsistentes | 9 formatos distintos mezclados |
| Timezones inconsistentes | ISO8601 con Z, +HH:MM, EST, sin tz, solo offset |
| Fechas imposibles | 9999-12-31, 0001-01-01, "ayer", "manyana" |
| Espacios | Leading, trailing, tabs, newlines |
| Caracteres especiales | , , Unicode |
| Duplicados | Filas exactas duplicadas |
| Nulos semanticos | 0, -1, 9999, 'N/A', 'Sin definir', '' mezclados |
| Autocontradiccion | activo=No pero fecha_futura; "aumento" con salario menor |
| IDs negativos | -5, -10 en TRANSACCIONES |
| Precision absurda | 20 decimales en montos |
| Tipos mezclados | $, EUR, "Gratis" en misma columna numerica |
| Booleanos inconsistentes | S/N, Y/N, 1/0, T/F, true/false mezclados |
| SQL injection | `<script>` tags en datos |
| Formato telefonos | 10+ formatos distintos (+34, (555), ext, etc.) |
| Multi-byte | Latin1 (ny,u) + CJK () + Emojis () mezclados |

## Fase 1  Completada 

### Lo que se logro

1. **De 5 a 23 tablas** cubriendo 20+ anti-patrones de esquema y ~17 de datos
2. **FK rotas**: 4 casos (FK sin PK destino, FK DISABLE, relacion fantasma, FK type mismatch)
3. **CHECK constraints contradictorias**: 4 constraints DISABLE que nunca podrian funcionar
4. **Indices redundantes**: 5 indices duplicados en EMPLEADOS
5. **Timezones inconsistentes**: 8 formatos de zona horaria mezclados en fechas
6. **Multi-byte**: Latin1 + CJK + Emojis mezclados en nombres y descripciones
7. **Tablas extremas**: 50 columnas (BACKUP_DATOS), 40 columnas (TABLA_BASE_DATOS), tipos deprecated (DATOS_MAESTROS)
8. **~5000+ filas totales** con densidad alta de corrupcion
9. **Tablas malas de verdad**: tipos incorrectos, defaults absurdos, columnas fantasma, palabras reservadas, CLOB/BLOB para flags, nombres monocolumna, etc.

### Pipeline de generacion validado

```
Docker Oracle 23c  ->  orchestrator.py  ->  23 tablas + ~5000 filas
                         |-- anti_patterns.py   (definiciones + datos)
                         |-- ddl_generator.py   (FK, CHECK DISABLE, indices)
                         \-- dml_generator.py   (INSERT con datos sucios)
```

### Listo para Fase 2

El pipeline Phase 2 (SchemaExtractor -> TextPreprocessor -> BERTEmbedder -> StructuralEncoder -> FeatureBuilder) ya puede consumir estas 23 tablas. Los modulos 6-10 estan pendientes.

### Modulos Phase 2 pendientes (6-10)
- [ ] **Modulo 6  DimensionalityReducer**: Reduccion de dimensionalidad (PCA/t-SNE/UMAP) a los vectores compuestos
- [ ] **Modulo 7  ClusterEngine**: Algoritmos de clustering sobre los vectores reducidos
- [ ] **Modulo 8  Evaluator**: Metricas de evaluacion de calidad del clustering
- [ ] **Modulo 9  Recommender**: Recomendacion de anti-patrones basada en clusters
- [ ] **Modulo 10  Visualizer**: Visualizacion de resultados (matricula de correlacion, scatter plots, etc.)
