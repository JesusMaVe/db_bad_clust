"""
anti_patterns.py — Documentation overlay for the synthetic anti-pattern schema.

The schema-generation machinery this file used to hold (AntiPatternTable,
generate_poorly_designed_tables, the row-data generators) had one consumer,
generation/schema_adapter.py, which itself had no callers once ground_truth.py
stopped deriving labels from the rule engine — see the `rule-engine` branch,
where that whole chain still lives. What's left here is the only part `main`
still uses: table/column comments applied to the live Oracle instance via
scripts/apply_comments.py, and read back by SchemaExtractor into the BERT
input text (see features/text_preprocessor.build_document).
"""

from __future__ import annotations

# Documentation overlay: partial, inconsistent comments like a real legacy DB.
# Applied to Oracle via apply_comments.py; extracted by SchemaExtractor and
# appended to the BERT input text (see text_preprocessor.process).
TABLE_COMMENTS: dict[str, str] = {
    "DEPARTAMENTOS": "Departamentos de la empresa. Heredada del sistema legacy de RRHH (1998).",
    "EMPLEADOS": "Empleados activos e inactivos. Fuente principal del módulo de RRHH.",
    "TBL_DATOS": "Tabla genérica de datos auxiliares del sistema antiguo. Nadie recuerda su propósito exacto.",
    "ORDENES_COMPRA": "Órdenes de compra del sistema de aprovisionamiento.",
    "CLIENTES_DIRECCIONES": "Clientes y sus direcciones de envío mezcladas en una sola tabla.",
    "PROVEEDORES": "Proveedores aprobados por el departamento de compras.",
    "PRODUCTOS": "Catálogo de productos con precios y stock mezclados en texto.",
    "TODO_EN_UNO": "Tabla polimórfica que guarda clientes, productos y empleados a la vez según TIPO_REGISTRO.",
    "AUDITORIA_LOG": "Log de auditoría de cambios sobre el resto de las tablas.",
    "CATEGORIAS": "Categorías jerárquicas de productos (árbol con niveles).",
    "TRANSACCIONES": "Movimientos financieros: cargos, abonos y transferencias.",
    "EMPLEADOS_HISTORIAL": "Historial de cambios salariales de los empleados.",
    "CONFIGURACION": "Parámetros de configuración del sistema en formato clave-valor (EAV).",
    "TABLA_BASE_DATOS": "Tabla de volcado genérico usada por el equipo de soporte para exportaciones.",
    "BACKUP_DATOS": "Copia de seguridad manual de varias tablas hecha en 2019. Nunca se limpió.",
    "DATOS_MAESTROS": "Datos maestros del sistema antiguo. Incluye tipos obsoletos LONG y LONG RAW.",
    "VENTAS": "Ventas por producto y vendedor. Alimentada por el POS de las tiendas.",
    "USUARIOS_WEB": "Usuarios del portal web con valores por defecto incoherentes.",
    "INVENTARIO": "Existencias de productos por almacén. Reemplazada por el WMS en 2022 pero aún recibe cargas.",
    "REPORTES": "Resultados precalculados de reportes. Las columnas usan palabras reservadas de SQL.",
    "METADATA": "Metadatos dinámicos definidos por el usuario. Todo guardado como CLOB/BLOB.",
    "TABLA_VACIA": "Tabla creada para una migración que se canceló. Nunca recibió datos.",
    "CACHE_TEMPORAL": "Caché de resultados pesados. Se suponía que se limpiaba cada noche.",
}

COLUMN_COMMENTS: dict[str, dict[str, str]] = {
    "DEPARTAMENTOS": {
        "ID": "Identificador del departamento. Secuencia compartida con la tabla antigua DEPT_OLD.",
        "PRESUPUESTO": "Presupuesto anual en miles de euros. Texto porque el sistema legacy no soportaba decimales.",
    },
    "EMPLEADOS": {
        "FECHA_NACIMIENTO": "Fecha de nacimiento en formato dd/mm/yyyy. Guardada como texto por compatibilidad con el formulario web antiguo.",
        "SALARIO": "Salario bruto anual en euros, incluye pagas extra.",
        "EMAIL": "Correo corporativo del empleado. La columna se creó por error como DATE y nunca se corrigió.",
        "ACTIVO": "Indicador de empleo vigente: Y/N, 1/0 o SI/NO según la época de la carga.",
        "COL_A_BORRAR_1": "Columna temporal de la migración de 2015. Pendiente de eliminar.",
        "DUPLICADO_NOMBRE": "Copia del nombre usada por un reporte de Excel que ya no existe.",
    },
    "TBL_DATOS": {
        "C1": "Posiblemente un identificador. Sin documentación.",
        "TEMP_DATA": "Datos temporales de depuración que nunca se borran.",
    },
    "ORDENES_COMPRA": {
        "FECHA_ORDEN": "Fecha de emisión de la orden. Formato inconsistente: dd/mm/yyyy, yyyy-mm-dd o texto libre.",
        "CANTIDAD": "Unidades solicitadas. Texto porque a veces incluye el sufijo aprox.",
        "PRECIO": "Precio unitario con impuestos incluidos.",
        "ESTADO": "Estado de la orden: 0=pendiente, 1=aprobada, 2=rechazada, 99=desconocido.",
        "TELF_CONTACTO": "Teléfono del destinatario. Formato libre, a veces incluye extensión.",
    },
    "CLIENTES_DIRECCIONES": {
        "TELEFONOS": "Todos los teléfonos del cliente concatenados en un solo campo, separados por comas o punto y coma.",
        "EMAILS": "Correos del cliente concatenados. Puede contener direcciones obsoletas.",
        "FAX": "Número de fax. Obsoleto desde 2010 pero se sigue mostrando en el reporte corporativo.",
    },
    "TODO_EN_UNO": {
        "TIPO_REGISTRO": "Discriminador polimórfico: CLI=cliente, PRO=producto, EMP=empleado, PED=pedido.",
        "NOMBRE_O_DESCRIPCION": "Nombre si es cliente o empleado; descripción si es producto.",
        "CANT_O_PRECIO": "Cantidad para pedidos, precio para productos, salario para empleados.",
        "FLAG_S_N": "Flag booleano genérico con valores S/N.",
    },
    "AUDITORIA_LOG": {
        "ACCION": "Operación ejecutada: INSERT, UPDATE, DELETE o TRUNCATE.",
        "FECHA": "Momento de la acción. Formato dd/mm/yyyy hh24:mi:ss como texto.",
        "REGISTRO_ID": "Identificador del registro afectado en la tabla TABLA_AFECTADA.",
    },
    "CATEGORIAS": {
        "CAT_PADRE_ID": "Categoría padre para la jerarquía de árbol. NULL en la raíz.",
        "NIVEL": "Profundidad en la jerarquía: 1, 2, 3.",
        "RUTA_COMPLETA": "Ruta materializada de la jerarquía: /Electronica/Portatiles/Ultrabooks.",
    },
    "TRANSACCIONES": {
        "MONTO": "Importe de la transacción en la moneda indicada por MONEDA.",
        "TIPO": "Tipo de movimiento: C=cargo, A=abono, T=transferencia.",
        "CONFIRMADO": "S=confirmada por el banco, N=pendiente.",
    },
    "EMPLEADOS_HISTORIAL": {
        "SALARIO_ANTERIOR": "Salario bruto anual antes del cambio.",
        "SALARIO_NUEVO": "Salario bruto anual después del cambio.",
        "TIPO_CAMBIO": "Motivo del cambio: promocion, ajuste, correccion.",
    },
    "CONFIGURACION": {
        "CLAVE": "Nombre del parámetro. Convención mixta: camelCase, SNAKE_CASE y kebab-case.",
        "VALOR": "Valor del parámetro. El tipo real se indica en TIPO_DATO.",
        "TIPO_DATO": "Tipo del valor: NUMERO, TEXTO, EMAIL, JSON, SECRETO, BASE64, COLOR.",
        "ULTIMA_MODIFICACION": "Fecha del último cambio al parámetro. Puede ser nunca.",
    },
    "VENTAS": {
        "PRODUCTO_FK": "SKU del producto vendido. Referencia a PRODUCTOS.SKU.",
        "PRECIO_TOTAL": "Importe total de la línea de venta, impuestos incluidos.",
        "VENDEDOR": "Nombre del vendedor que realizó la venta.",
    },
    "USUARIOS_WEB": {
        "INTENTOS_FALLIDOS": "Contador de intentos de login fallidos. El valor por defecto NADA es un error.",
        "ROL": "Rol del usuario: admin, editor, visitante. El valor por defecto 12345 es un error.",
        "FECHA_EXPIRACION": "Fecha de expiración de la cuenta. Por defecto 9999-12-31 significa nunca.",
    },
    "INVENTARIO": {
        "STOCK": "Existencias actuales en el almacén principal.",
        "STOCK_2": "Segundo conteo de existencias hecho en el inventario físico. Texto.",
        "STOCK_COPIA": "Copia del stock hecha por el procedimiento de sincronización nocturno.",
    },
    "REPORTES": {
        "NULL": "Comentario libre del reporte. La columna usa una palabra reservada de SQL.",
        "SELECT": "Consulta SQL que generó el reporte. Palabra reservada.",
    },
    "METADATA": {
        "NOMBRE_CAMPO": "Nombre del campo dinámico definido por el usuario.",
        "VALOR_CAMPO": "Valor serializado en binario del campo dinámico.",
        "ES_ACTIVO": "Indica si el campo dinámico sigue en uso: S/N.",
    },
    "CACHE_TEMPORAL": {
        "DATOS": "Payload original de la petición cacheada, serializado en binario.",
        "TIMESTAMP": "Momento de creación de la entrada de caché como texto.",
    },
}


