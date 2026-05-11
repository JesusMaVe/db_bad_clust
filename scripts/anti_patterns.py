# scripts/anti_patterns.py
"""
Catálogo de malas prácticas de base de datos para generar datos de entrenamiento.

Este módulo define anti-patrones comunes en diseño de bases de datos:
- Tablas sin claves primarias
- Nombres de columna inconsistentes
- Tipos de datos incorrectos
- Datos duplicados
- Valores nulos inapropiados
- Formatos de fecha inconsistentes
- HTML/XML en campos de texto
- Espacios en blanco sobrantes
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

@dataclass
class AntiPatternTable:
    """Define una tabla con problemas de diseño intencionales."""
    name: str
    columns: Dict[str, str]  # columna -> tipo SQL
    # Problemas comunes a nivel tabla
    no_primary_key: bool = False
    inconsistent_naming: bool = False  # mezcla camelCase, snake_case, espacios
    reserved_word_columns: bool = False  # columnas con nombres problemáticos
    wrong_data_types: bool = False  # tipos de dato incorrectos para los datos

@dataclass
class AntiPatternData:
    """Define problemas en los datos INSERT."""
    table_name: str
    rows: List[Dict[str, Any]]
    # Problemas comunes en datos
    nulls_in_required: bool = False
    inconsistent_date_formats: bool = False
    duplicate_rows: bool = False
    html_in_text: bool = False
    leading_trailing_spaces: bool = False
    special_characters: bool = False
    wrong_data_types: bool = False

# --- Datos de ejemplo para poblar las tablas ---

SAMPLE_NAMES = [
    "Juan Pérez", "María García", "Carlos López", "Ana Martínez",
    "Pedro Rodríguez", "Laura Hernández", "José González", "Sofía Díaz",
    "Miguel Torres", "Isabel Ramírez", "Francisco Flores", "Carmen Ruiz",
    "Alejandro Vargas", "Patricia Jiménez", "Roberto Castro", "Gabriela Ortiz"
]

SAMPLE_DEPARTMENTS = [
    "Ventas", "Marketing", "IT", "RRHH", "Finanzas",
    "Operaciones", "Legal", "Desarrollo", "Soporte", "Dirección"
]

SAMPLE_PRODUCTS = [
    "Laptop HP", "Monitor Dell", "Teclado Mecánico", "Mouse Inalámbrico",
    "Impresora Laser", "Disco Duro 1TB", "Memoria RAM 16GB", "Webcam HD",
    "Auriculares Bluetooth", "Hub USB-C", "Cable HDMI", "Adaptador VGA"
]

SAMPLE_CITIES = [
    "Madrid", "Barcelona", "Valencia", "Sevilla", "Bilbao",
    "Málaga", "Zaragoza", "Murcia", "Palma", "Las Palmas"
]

# --- Generadores de Malas Prácticas ---

def _generate_inconsistent_date() -> str:
    """Genera fechas en formatos inconsistentes como anti-patrón."""
    base_date = datetime.now() - timedelta(days=random.randint(1, 365))
    formats = [
        base_date.strftime("%Y-%m-%d"),              # 2024-01-15
        base_date.strftime("%d/%m/%Y"),              # 15/01/2024
        base_date.strftime("%m-%d-%Y"),              # 01-15-2024
        base_date.strftime("%d-%b-%Y"),              # 15-Jan-2024
        base_date.strftime("%B %d, %Y"),             # January 15, 2024
        base_date.strftime("%Y%m%d"),                # 20240115
        "Fecha desconocida",                          # Texto en campo fecha
        None,                                        # Nulo en campo fecha
        "",                                          # Vacío en campo fecha
    ]
    return random.choice(formats)

def _generate_bad_text(original: str) -> str:
    """Genera texto con problemas comunes."""
    problems = [
        f"<b>{original}</b>",                        # HTML
        f"<p>{original}</p>",                        # HTML
        f"  {original}  ",                           # Espacios
        f"\t{original}\t",                           # Tabulaciones
        f"{original}\n",                             # Nueva línea
        original.upper(),                            # Todo mayúsculas
        original.lower(),                            # Todo minúsculas
        f"{original}®™",                             # Caracteres especiales
        None,                                        # Nulo
        "",                                          # Vacío
    ]
    return random.choice(problems)

def generate_poorly_designed_tables() -> List[AntiPatternTable]:
    """
    Genera una lista de definiciones de tablas con mal diseño.
    
    Returns:
        Lista de AntiPatternTable con diferentes problemas de diseño
    """
    tables = []

    # 1. Tabla de empleados sin clave primaria, tipos incorrectos
    tables.append(AntiPatternTable(
        name="EMPLEADOS",
        columns={
            "ID": "NUMBER",
            "NOMBRE_EMPLEADO": "VARCHAR2(100)",
            "FECHA_NACIMIENTO": "VARCHAR2(20)",  # Fecha como texto
            "SALARIO": "VARCHAR2(50)",           # Número como texto
            "DEPARTAMENTO": "NUMBER",            # Texto como número
            "EMAIL": "DATE",                     # Email como fecha
            "ACTIVO": "VARCHAR2(10)"             # Booleano como texto
        },
        no_primary_key=True,
        wrong_data_types=True,
        inconsistent_naming=False  # Usamos nombres limpios pero con tipos malos
    ))

    # 2. Tabla con nombres crípticos y sin documentación
    tables.append(AntiPatternTable(
        name="TBL_DATOS",
        columns={
            "C1": "NUMBER",
            "C2": "VARCHAR2(255)",
            "C3": "VARCHAR2(255)",
            "C4": "VARCHAR2(255)",
            "C5": "VARCHAR2(20)",
            "C6": "DATE"
        },
        no_primary_key=True,
        inconsistent_naming=True  # Nombres no descriptivos
    ))

    # 3. Tabla de órdenes con problemas de diseño
    tables.append(AntiPatternTable(
        name="ORDENES_COMPRA",
        columns={
            "ORDER_ID": "NUMBER",
            "FECHA_ORDEN": "VARCHAR2(50)",      # Fecha como texto largo
            "CLIENTE": "VARCHAR2(100)",
            "PRODUCTO": "VARCHAR2(200)",
            "CANTIDAD": "VARCHAR2(10)",          # Número como texto
            "PRECIO": "VARCHAR2(20)",            # Precio como texto
            "NOTAS": "CLOB",                     # Campo grande para texto corto
            "ESTADO": "NUMBER"                   # Estado como número
        },
        no_primary_key=True,
        wrong_data_types=True
    ))

    # 4. Tabla con datos redundantes y sin normalizar
    tables.append(AntiPatternTable(
        name="CLIENTES_DIRECCIONES",
        columns={
            "ID_CLIENTE": "NUMBER",
            "NOMBRE_CLIENTE": "VARCHAR2(100)",
            "DIRECCION_COMPLETA": "VARCHAR2(500)",  # Todo en un campo
            "TELEFONOS": "VARCHAR2(200)",           # Múltiples teléfonos juntos
            "EMAILS": "VARCHAR2(300)",              # Múltiples emails juntos
            "PROVINCIA_CIUDAD": "VARCHAR2(100)",    # Provincia y ciudad juntos
            "CODIGO_POSTAL_PAIS": "VARCHAR2(50)"    # CP y país juntos
        },
        no_primary_key=True,
        inconsistent_naming=True
    ))

    # 5. Tabla con columnas que almacenan múltiples valores
    tables.append(AntiPatternTable(
        name="PRODUCTOS",
        columns={
            "SKU": "VARCHAR2(20)",
            "DESCRIPCION": "VARCHAR2(500)",
            "PRECIOS": "VARCHAR2(200)",         # Lista de precios separados por coma
            "CATEGORIAS": "VARCHAR2(200)",      # Múltiples categorías juntas
            "PROVEEDORES_ID": "VARCHAR2(100)",  # Lista de IDs de proveedores
            "TALLAS_DISPONIBLES": "VARCHAR2(50)" # "S,M,L,XL" en un campo
        },
        no_primary_key=True
    ))

    return tables

def generate_bad_data_for_table(table: AntiPatternTable) -> AntiPatternData:
    """
    Genera datos INSERT con problemas para una tabla específica.
    
    Args:
        table: Definición de la tabla con anti-patrones
        
    Returns:
        AntiPatternData con filas que contienen malas prácticas
    """
    rows = []
    num_rows = random.randint(8, 20)
    
    if table.name == "EMPLEADOS":
        rows = _generate_bad_empleados_data(num_rows)
    elif table.name == "TBL_DATOS":
        rows = _generate_bad_tbl_datos_data(num_rows)
    elif table.name == "ORDENES_COMPRA":
        rows = _generate_bad_ordenes_data(num_rows)
    elif table.name == "CLIENTES_DIRECCIONES":
        rows = _generate_bad_clientes_data(num_rows)
    elif table.name == "PRODUCTOS":
        rows = _generate_bad_productos_data(num_rows)
    
    return AntiPatternData(
        table_name=table.name,
        rows=rows,
        nulls_in_required=True,
        inconsistent_date_formats=True,
        duplicate_rows=random.choice([True, False]),
        html_in_text=True,
        leading_trailing_spaces=True,
        special_characters=True,
        wrong_data_types=table.wrong_data_types
    )

def _generate_bad_empleados_data(num_rows: int) -> List[Dict[str, Any]]:
    """Genera datos malos para la tabla EMPLEADOS."""
    rows = []
    for i in range(1, num_rows + 1):
        rows.append({
            "ID": i if random.random() > 0.3 else None,  # IDs nulos a veces
            "NOMBRE_EMPLEADO": _generate_bad_text(random.choice(SAMPLE_NAMES)),
            "FECHA_NACIMIENTO": _generate_inconsistent_date(),
            "SALARIO": random.choice([  # Salario en formato inconsistente
                f"{random.randint(20000, 80000)}",
                f"${random.randint(20000, 80000)}",
                f"{random.randint(20000, 80000)} euros",
                None,
                "Sin definir"
            ]),
            "DEPARTAMENTO": random.choice([  # Departamento como número (mal)
                str(random.randint(1, 10)),
                random.choice(SAMPLE_DEPARTMENTS),  # A veces texto
                None
            ]),
            "EMAIL": _generate_inconsistent_date(),  # Email como fecha (mal)
            "ACTIVO": random.choice(["Sí", "No", "1", "0", "true", "false", None])
        })
    return rows

def _generate_bad_tbl_datos_data(num_rows: int) -> List[Dict[str, Any]]:
    """Genera datos malos para la tabla TBL_DATOS."""
    rows = []
    for i in range(1, num_rows + 1):
        rows.append({
            "C1": i if random.random() > 0.4 else None,
            "C2": _generate_bad_text(f"Dato_{i}") if random.random() > 0.2 else None,
            "C3": random.choice([
                _generate_inconsistent_date(),
                f"ID_{random.randint(1, 100)}",
                str(random.randint(1000, 9999)),
                None,
                "<data>XML content</data>"
            ]),
            "C4": random.choice([
                random.choice(SAMPLE_CITIES),
                random.choice(SAMPLE_PRODUCTS),
                str(random.randint(1, 100)),
                None,
                "  datos con espacios  "
            ]),
            "C5": random.choice(["Alta", "Media", "Baja", "1", "2", "3", None]),
            "C6": _generate_inconsistent_date() if random.random() > 0.3 else None
        })
    return rows

def _generate_bad_ordenes_data(num_rows: int) -> List[Dict[str, Any]]:
    """Genera datos malos para la tabla ORDENES_COMPRA."""
    rows = []
    for i in range(1, num_rows + 1):
        rows.append({
            "ORDER_ID": i if random.random() > 0.2 else None,
            "FECHA_ORDEN": _generate_inconsistent_date(),
            "CLIENTE": _generate_bad_text(random.choice(SAMPLE_NAMES)),
            "PRODUCTO": random.choice([
                _generate_bad_text(random.choice(SAMPLE_PRODUCTS)),
                None,
                "",
                "Producto sin especificar"
            ]),
            "CANTIDAD": random.choice([
                str(random.randint(1, 50)),
                f"{random.randint(1, 50)} unidades",
                "varios",
                None,
                "N/A"
            ]),
            "PRECIO": random.choice([
                f"${random.randint(10, 1000)}",
                f"{random.randint(10, 1000)}€",
                str(random.randint(10, 1000)),
                "Gratis",
                None
            ]),
            "NOTAS": random.choice([
                "<p>Orden urgente</p>",
                "  Entregar en recepción  ",
                "Cliente VIP®™",
                None,
                "",
                "Llamar antes de entregar\nTel: 555-0123"
            ]),
            "ESTADO": random.choice([1, 2, 3, None, "pendiente", "completado"])
        })
    
    # Agregar algunas filas duplicadas intencionalmente
    if rows:
        rows.append(rows[0].copy())  # Duplicado exacto
        rows.append(rows[1].copy())  # Otro duplicado
    
    return rows

def _generate_bad_clientes_data(num_rows: int) -> List[Dict[str, Any]]:
    """Genera datos malos para la tabla CLIENTES_DIRECCIONES."""
    rows = []
    for i in range(1, num_rows + 1):
        rows.append({
            "ID_CLIENTE": i if random.random() > 0.3 else None,
            "NOMBRE_CLIENTE": _generate_bad_text(random.choice(SAMPLE_NAMES)),
            "DIRECCION_COMPLETA": random.choice([
                f"Calle {random.choice(SAMPLE_CITIES)} #{random.randint(1, 200)}, Piso {random.randint(1, 10)}",
                f"Av. Principal {random.randint(1, 100)}, {random.choice(SAMPLE_CITIES)}",
                None,
                "Sin dirección registrada"
            ]),
            "TELEFONOS": random.choice([
                f"555-{random.randint(1000, 9999)}, 555-{random.randint(1000, 9999)}",
                f"+34 {random.randint(600000000, 699999999)}",
                None,
                "No disponible"
            ]),
            "EMAILS": random.choice([
                f"cliente{i}@email.com, cliente{i}_alt@email.com",
                f"{random.choice(SAMPLE_NAMES).lower().replace(' ', '.')}@empresa.com",
                None,
                ""
            ]),
            "PROVINCIA_CIUDAD": f"{random.choice(SAMPLE_CITIES)}, {random.choice(SAMPLE_CITIES)}",
            "CODIGO_POSTAL_PAIS": f"{random.randint(10000, 99999)}-{random.choice(['España', 'México', 'Argentina', 'Colombia'])}"
        })
    return rows

def _generate_bad_productos_data(num_rows: int) -> List[Dict[str, Any]]:
    """Genera datos malos para la tabla PRODUCTOS."""
    rows = []
    for i in range(1, num_rows + 1):
        rows.append({
            "SKU": f"SKU-{random.randint(1000, 9999)}" if random.random() > 0.2 else None,
            "DESCRIPCION": _generate_bad_text(random.choice(SAMPLE_PRODUCTS)),
            "PRECIOS": random.choice([
                f"19.99, 24.99, 29.99",
                f"${random.randint(10, 100)}",
                f"{random.randint(10, 500)}",
                None,
                "Consultar"
            ]),
            "CATEGORIAS": random.choice([
                f"Electrónica, Oficina, Hogar",
                f"{random.choice(['Tecnología', 'Accesorios', 'Periféricos'])}",
                None,
                ""
            ]),
            "PROVEEDORES_ID": random.choice([
                f"PROV_{random.randint(1, 10)}, PROV_{random.randint(11, 20)}",
                str(random.randint(1, 50)),
                None
            ]),
            "TALLAS_DISPONIBLES": random.choice([
                "S,M,L,XL",
                "Única",
                "38,40,42,44",
                None,
                "N/A"
            ])
        })
    return rows
