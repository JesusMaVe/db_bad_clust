import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

@dataclass
class AntiPatternTable:
    name: str
    columns: Dict[str, str]
    no_primary_key: bool = False
    inconsistent_naming: bool = False
    reserved_word_columns: bool = False
    wrong_data_types: bool = False
    self_referencing: bool = False
    polymorphic: bool = False
    giant_table: bool = False
    eav_antipattern: bool = False
    impossible_data: bool = False
    self_contradictory: bool = False
    fk_constraints: Dict[str, str] = field(default_factory=dict)
    check_constraints: List[str] = field(default_factory=list)
    indexes: List[List[str]] = field(default_factory=list)

@dataclass
class AntiPatternData:
    table_name: str
    rows: List[Dict[str, Any]]
    nulls_in_required: bool = False
    inconsistent_date_formats: bool = False
    duplicate_rows: bool = False
    html_in_text: bool = False
    leading_trailing_spaces: bool = False
    special_characters: bool = False
    wrong_data_types: bool = False
    self_contradictory: bool = False
    semantic_nulls: bool = False
    impossible_dates: bool = False
    negative_ids: bool = False

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

SAMPLE_PAYMENT_METHODS = [
    "Tarjeta", "Efectivo", "Transferencia", "PayPal", "Bitcoin",
    "Contra reembolso", "CHEQUE", "wallet", "CRYPTO", "Crédito"
]

SAMPLE_CURRENCIES = [
    "USD", "EUR", "MXN", "dólares", "euros", "pesos", "$", "$USD", None, ""
]

SAMPLE_PHONE_FORMATS = [
    "555-1234",
    "+34 612 345 678",
    "(555) 123-4567",
    "555.123.4567",
    "612345678",
    "+1-555-123-4567",
    "555 1234 ext 101",
    None,
    "No disponible",
    ""
]

SAMPLE_CATEGORIES = [
    "Electrónica", "Oficina", "Hogar", "Deportes", "Juguetes",
    "Ropa", "Alimentos", "Libros", "Música", "Jardinería"
]

SAMPLE_MULTIBYTE = [
    "Café 🚀 データ",
    "Ñoño y Çiğköfte",
    "システム管理 🎉",
    "Straße über München",
    "Français Español Português ✅",
    "データベース 💾 Fácil",
    "Käse 🧀 Güte ✨",
    "Téli este — különleges 💫",
    "中文 混合 with Latin1 üñí",
    "普通話 / Español / 日本語",
]

def _generate_inconsistent_date() -> str:
    base_date = datetime.now() - timedelta(days=random.randint(1, 365))
    formats = [
        base_date.strftime("%Y-%m-%d"),
        base_date.strftime("%d/%m/%Y"),
        base_date.strftime("%m-%d-%Y"),
        base_date.strftime("%d-%b-%Y"),
        base_date.strftime("%B %d, %Y"),
        base_date.strftime("%Y%m%d"),
        "Fecha desconocida",
        None,
        "",
    ]
    return random.choice(formats)

def _generate_impossible_date() -> str:
    formats = [
        "9999-12-31",
        "0001-01-01",
        "31-31-2024",
        "2024-13-01",
        "2024-00-15",
        "Fecha inválida",
        "ayer",
        "mañana",
        None,
        "",
        datetime.now().strftime("%Y-%m-%d"),
    ]
    return random.choice(formats)

def _generate_timezone_date() -> str:
    base = datetime.now() - timedelta(days=random.randint(1, 365))
    h, m = random.randint(0, 23), random.randint(0, 59)
    tz_offset = f"{'+' if random.random() > 0.3 else '-'}{random.randint(0, 12):02d}:{random.choice(['00', '30', '45'])}"
    formats = [
        base.strftime("%Y-%m-%dT%H:%M:%SZ"),
        f"{base.strftime('%Y-%m-%d %H:%M:%S')} {tz_offset}",
        base.strftime("%Y-%m-%d %H:%M:%S"),
        f"{base.strftime('%d/%m/%Y')} {h}:{m:02d} AM EST",
        f"{base.strftime('%B %d, %Y %H:%M:%S.%f')}",
        f"{base.strftime('%Y-%m-%dT%H:%M:%S')}.{random.randint(100, 999)}{tz_offset}",
        tz_offset,
        "2024-01-15T10:30:00.123+05:30",
        None,
        "",
    ]
    return random.choice(formats)

def _generate_bad_text(original: str) -> str:
    problems = [
        f"<b>{original}</b>",
        f"<p>{original}</p>",
        f"  {original}  ",
        f"\t{original}\t",
        f"{original}\n",
        original.upper(),
        original.lower(),
        f"{original}®™",
        None,
        "",
    ]
    return random.choice(problems)

def generate_poorly_designed_tables() -> List[AntiPatternTable]:
    tables = []

    tables.append(AntiPatternTable(
        name="DEPARTAMENTOS",
        columns={
            "ID": "NUMBER",
            "NOMBRE_DEPARTAMENTO": "VARCHAR2(100)",
            "PRESUPUESTO": "VARCHAR2(20)",
        },
        no_primary_key=True,
        wrong_data_types=True,
    ))

    tables.append(AntiPatternTable(
        name="EMPLEADOS",
        columns={
            "ID": "NUMBER",
            "NOMBRE_EMPLEADO": "VARCHAR2(100)",
            "FECHA_NACIMIENTO": "VARCHAR2(20)",
            "SALARIO": "VARCHAR2(50)",
            "DEPARTAMENTO": "NUMBER",
            "EMAIL": "DATE",
            "ACTIVO": "VARCHAR2(10)",
            "COL_A_BORRAR_1": "VARCHAR2(4000)",
            "DUPLICADO_NOMBRE": "VARCHAR2(100)",
            "FECHA_ALTA": "VARCHAR2(20)",
        },
        no_primary_key=True,
        wrong_data_types=True,
        fk_constraints={
            "DEPARTAMENTO": "DEPARTAMENTOS(ID) ON DELETE CASCADE",
        },
        check_constraints=[
            "SALARIO > 0 AND SALARIO < 0",
        ],
        indexes=[
            ["ID"], ["ID"], ["ID"],
            ["NOMBRE_EMPLEADO"], ["NOMBRE_EMPLEADO"],
        ],
    ))

    tables.append(AntiPatternTable(
        name="TBL_DATOS",
        columns={
            "C1": "NUMBER",
            "C2": "VARCHAR2(255)",
            "C3": "VARCHAR2(255)",
            "C4": "VARCHAR2(255)",
            "C5": "VARCHAR2(20)",
            "C6": "DATE",
            "C7": "VARCHAR2(4000)",
            "COL_EXTRA": "VARCHAR2(255)",
            "TEMP_DATA": "CLOB",
        },
        no_primary_key=True,
        inconsistent_naming=True,
        indexes=[
            ["C1"],
        ],
    ))

    tables.append(AntiPatternTable(
        name="ORDENES_COMPRA",
        columns={
            "ORDER_ID": "NUMBER",
            "FECHA_ORDEN": "VARCHAR2(50)",
            "CLIENTE": "VARCHAR2(100)",
            "PRODUCTO": "VARCHAR2(200)",
            "CANTIDAD": "VARCHAR2(10)",
            "PRECIO": "VARCHAR2(20)",
            "NOTAS": "CLOB",
            "ESTADO": "NUMBER",
            "DIRECCION_ENVIO": "VARCHAR2(4000)",
            "TELF_CONTACTO": "VARCHAR2(4000)",
            "BACKUP_ORDER_ID": "NUMBER",
        },
        no_primary_key=True,
        wrong_data_types=True,
        indexes=[
            ["ORDER_ID"],
            ["CLIENTE"],
        ],
    ))

    tables.append(AntiPatternTable(
        name="CLIENTES_DIRECCIONES",
        columns={
            "ID_CLIENTE": "NUMBER",
            "NOMBRE_CLIENTE": "VARCHAR2(100)",
            "DIRECCION_COMPLETA": "VARCHAR2(500)",
            "TELEFONOS": "VARCHAR2(200)",
            "EMAILS": "VARCHAR2(300)",
            "PROVINCIA_CIUDAD": "VARCHAR2(100)",
            "CODIGO_POSTAL_PAIS": "VARCHAR2(50)",
            "FAX": "VARCHAR2(50)",
            "DIRECCION_ALTERNATIVA": "VARCHAR2(500)",
            "OBSERVACIONES": "CLOB",
        },
        no_primary_key=True,
        inconsistent_naming=True,
    ))

    tables.append(AntiPatternTable(
        name="PROVEEDORES",
        columns={
            "ID": "NUMBER",
            "NOMBRE_PROVEEDOR": "VARCHAR2(100)",
            "CONTACTO": "VARCHAR2(100)",
            "TELEFONO": "VARCHAR2(50)",
        },
        no_primary_key=True,
    ))

    tables.append(AntiPatternTable(
        name="PRODUCTOS",
        columns={
            "SKU": "VARCHAR2(20)",
            "DESCRIPCION": "VARCHAR2(500)",
            "PRECIOS": "VARCHAR2(200)",
            "CATEGORIAS": "VARCHAR2(200)",
            "PROVEEDORES_ID": "VARCHAR2(100)",
            "TALLAS_DISPONIBLES": "VARCHAR2(50)",
            "PROVEEDOR_NOMBRE": "VARCHAR2(4000)",
            "STOCK_SEGURIDAD": "VARCHAR2(20)",
            "UBICACION_ALMACEN": "VARCHAR2(4000)",
        },
        no_primary_key=True,
        check_constraints=[
            "PRECIOS IS NOT NULL",
        ],
    ))

    tables.append(AntiPatternTable(
        name="TODO_EN_UNO",
        columns={
            "ID": "NUMBER",
            "TIPO_REGISTRO": "VARCHAR2(20)",
            "NOMBRE_O_DESCRIPCION": "VARCHAR2(4000)",
            "CANT_O_PRECIO": "VARCHAR2(4000)",
            "FECHA_O_DIRECCION": "VARCHAR2(4000)",
            "ESTADO_O_ACTIVO": "VARCHAR2(4000)",
            "OBSERVACIONES": "VARCHAR2(4000)",
            "FLAG_S_N": "CHAR(1)",
            "FLAG_Y_N": "CHAR(1)",
            "FLAG_1_0": "VARCHAR2(3)",
            "FLAG_T_F": "CHAR(1)",
            "EXTRA_INFO": "VARCHAR2(4000)",
            "COL_A_BORRAR": "VARCHAR2(4000)",
            "BACKUP_COLUMN": "VARCHAR2(4000)",
            "REPETIDA_1": "VARCHAR2(4000)",
            "FECHA_CREACION": "DATE",
        },
        no_primary_key=True,
        polymorphic=True,
        giant_table=True,
        wrong_data_types=True,
    ))

    tables.append(AntiPatternTable(
        name="AUDITORIA_LOG",
        columns={
            "ID": "NUMBER",
            "TABLA_AFECTADA": "VARCHAR2(50)",
            "REGISTRO_ID": "NUMBER",
            "ACCION": "VARCHAR2(20)",
            "USUARIO": "VARCHAR2(50)",
            "FECHA": "VARCHAR2(30)",
            "DETALLE": "VARCHAR2(4000)",
        },
        no_primary_key=True,
        fk_constraints={
            "REGISTRO_ID": "TODO_EN_UNO(ID) DISABLE",
        },
    ))

    tables.append(AntiPatternTable(
        name="CATEGORIAS",
        columns={
            "ID": "NUMBER",
            "NOMBRE": "VARCHAR2(100)",
            "CAT_PADRE_ID": "NUMBER",
            "NIVEL": "VARCHAR2(10)",
            "ACTIVO": "CHAR(1)",
            "RUTA_COMPLETA": "VARCHAR2(1000)",
        },
        no_primary_key=True,
        self_referencing=True,
        wrong_data_types=True,
        check_constraints=[
            "ACTIVO IN ('S', 'N')",
        ],
    ))

    tables.append(AntiPatternTable(
        name="TRANSACCIONES",
        columns={
            "ID": "NUMBER",
            "DESCRIPCION": "VARCHAR2(200)",
            "MONTO": "VARCHAR2(50)",
            "FECHA": "VARCHAR2(30)",
            "TIPO": "CHAR(1)",
            "MONEDA": "VARCHAR2(20)",
            "CONFIRMADO": "CHAR(1)",
        },
        no_primary_key=True,
        impossible_data=True,
        wrong_data_types=True,
        check_constraints=[
            "MONTO >= 0",
        ],
    ))

    tables.append(AntiPatternTable(
        name="EMPLEADOS_HISTORIAL",
        columns={
            "ID": "NUMBER",
            "EMPLEADO_ID": "NUMBER",
            "SALARIO_ANTERIOR": "VARCHAR2(20)",
            "SALARIO_NUEVO": "VARCHAR2(20)",
            "FECHA_CAMBIO": "VARCHAR2(30)",
            "TIPO_CAMBIO": "VARCHAR2(30)",
            "OBSERVACIONES": "VARCHAR2(500)",
        },
        no_primary_key=True,
        self_contradictory=True,
        wrong_data_types=True,
    ))

    tables.append(AntiPatternTable(
        name="CONFIGURACION",
        columns={
            "ID": "NUMBER",
            "CLAVE": "VARCHAR2(100)",
            "VALOR": "VARCHAR2(4000)",
            "TIPO_DATO": "VARCHAR2(20)",
            "ACTIVO": "CHAR(1)",
            "ULTIMA_MODIFICACION": "VARCHAR2(30)",
        },
        no_primary_key=True,
        eav_antipattern=True,
    ))

    return tables

def generate_bad_data_for_table(table: AntiPatternTable) -> AntiPatternData:
    rows = []
    num_rows = random.randint(10, 25)

    generators = {
        "DEPARTAMENTOS": _generate_bad_departamentos_data,
        "EMPLEADOS": _generate_bad_empleados_data,
        "TBL_DATOS": _generate_bad_tbl_datos_data,
        "ORDENES_COMPRA": _generate_bad_ordenes_data,
        "CLIENTES_DIRECCIONES": _generate_bad_clientes_data,
        "PROVEEDORES": _generate_bad_proveedores_data,
        "PRODUCTOS": _generate_bad_productos_data,
        "TODO_EN_UNO": _generate_bad_todo_en_uno_data,
        "AUDITORIA_LOG": _generate_bad_auditoria_data,
        "CATEGORIAS": _generate_bad_categorias_data,
        "TRANSACCIONES": _generate_bad_transacciones_data,
        "EMPLEADOS_HISTORIAL": _generate_bad_empleados_historial_data,
        "CONFIGURACION": _generate_bad_configuracion_data,
    }

    gen = generators.get(table.name)
    if gen:
        rows = gen(num_rows)

    return AntiPatternData(
        table_name=table.name,
        rows=rows,
        nulls_in_required=True,
        inconsistent_date_formats=True,
        duplicate_rows=random.choice([True, False]),
        html_in_text=True,
        leading_trailing_spaces=True,
        special_characters=True,
        wrong_data_types=table.wrong_data_types,
        self_contradictory=table.self_contradictory,
        impossible_dates=table.impossible_data,
        negative_ids=table.impossible_data,
        semantic_nulls=True,
    )

def _generate_bad_departamentos_data(num_rows: int) -> List[Dict[str, Any]]:
    rows = []
    for i in range(1, num_rows + 1):
        rows.append({
            "ID": i,
            "NOMBRE_DEPARTAMENTO": random.choice([
                random.choice(SAMPLE_DEPARTMENTS),
                _generate_bad_text(random.choice(SAMPLE_DEPARTMENTS)),
                random.choice(SAMPLE_MULTIBYTE),
                None,
            ]),
            "PRESUPUESTO": random.choice([
                f"{random.randint(50000, 500000)}",
                f"${random.randint(50000, 500000)}",
                None,
                "Sin presupuesto",
            ]),
        })
    return rows

def _generate_bad_proveedores_data(num_rows: int) -> List[Dict[str, Any]]:
    rows = []
    for i in range(1, num_rows + 1):
        rows.append({
            "ID": i,
            "NOMBRE_PROVEEDOR": random.choice([
                f"Proveedor {random.choice(SAMPLE_CITIES)}",
                _generate_bad_text(f"Supply {random.choice(SAMPLE_CITIES)}"),
                random.choice(SAMPLE_MULTIBYTE),
                None,
            ]),
            "CONTACTO": random.choice([
                f"contacto{i}@proveedor.com",
                _generate_bad_text(random.choice(SAMPLE_NAMES)),
                None,
            ]),
            "TELEFONO": random.choice(SAMPLE_PHONE_FORMATS),
        })
    return rows

def _generate_bad_auditoria_data(num_rows: int) -> List[Dict[str, Any]]:
    rows = []
    for i in range(1, num_rows + 1):
        rows.append({
            "ID": i if random.random() > 0.1 else None,
            "TABLA_AFECTADA": random.choice([
                "EMPLEADOS", "PRODUCTOS", "TODO_EN_UNO", None, "",
            ]),
            "REGISTRO_ID": random.choice([
                random.randint(1, 100),
                -1,
                99999,
                None,
            ]),
            "ACCION": random.choice([
                "INSERT", "UPDATE", "DELETE", "SELECT", None, "",
            ]),
            "USUARIO": random.choice([
                random.choice(SAMPLE_NAMES),
                "admin", "root", "sys", None, "",
            ]),
            "FECHA": _generate_timezone_date(),
            "DETALLE": random.choice([
                _generate_bad_text(f"Cambio en registro {i}"),
                None,
                "",
                "<script>audit</script>",
                random.choice(SAMPLE_MULTIBYTE),
            ]),
        })
    return rows

def _generate_bad_empleados_data(num_rows: int) -> List[Dict[str, Any]]:
    rows = []
    for i in range(1, num_rows + 1):
        activo = random.choice(["Sí", "No", "1", "0", "true", "false", None])
        fecha_alta = _generate_inconsistent_date()
        nombre = _generate_bad_text(random.choice(SAMPLE_NAMES))

        contradictory = random.random() < 0.2
        if contradictory and activo in ("No", "0", "false"):
            fecha_alta = (datetime.now() + timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d")

        rows.append({
            "ID": i if random.random() > 0.3 else None,
            "NOMBRE_EMPLEADO": nombre,
            "FECHA_NACIMIENTO": _generate_inconsistent_date(),
            "SALARIO": random.choice([
                f"{random.randint(20000, 80000)}",
                f"${random.randint(20000, 80000)}",
                f"{random.randint(20000, 80000)} euros",
                None,
                "Sin definir",
            ]),
            "DEPARTAMENTO": random.choice([
                str(random.randint(1, 10)),
                random.choice(SAMPLE_DEPARTMENTS),
                None,
            ]),
            "EMAIL": _generate_inconsistent_date(),
            "ACTIVO": activo,
            "FECHA_ALTA": fecha_alta,
        })
    return rows

def _generate_bad_tbl_datos_data(num_rows: int) -> List[Dict[str, Any]]:
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
                "<data>XML content</data>",
            ]),
            "C4": random.choice([
                random.choice(SAMPLE_CITIES),
                random.choice(SAMPLE_PRODUCTS),
                str(random.randint(1, 100)),
                None,
                "  datos con espacios  ",
            ]),
            "C5": random.choice(["Alta", "Media", "Baja", "1", "2", "3", None]),
            "C6": _generate_inconsistent_date() if random.random() > 0.3 else None,
            "C7": random.choice([
                f"VALOR_{i}",
                None,
                "",
                "<script>alert('xss')</script>",
                "  " * 10,
            ]),
            "TEMP_DATA": random.choice([
                f"<root><item id='{i}'/><root>",
                None,
                "  ",
                "TEMP_DATA_" + str(random.randint(1, 1000)),
            ]),
        })
    return rows

def _generate_bad_ordenes_data(num_rows: int) -> List[Dict[str, Any]]:
    rows = []
    for i in range(1, num_rows + 1):
        telefono = random.choice(SAMPLE_PHONE_FORMATS)
        if telefono is not None:
            telefono = _generate_bad_text(str(telefono))

        rows.append({
            "ORDER_ID": i if random.random() > 0.2 else None,
            "FECHA_ORDEN": random.choice([
                _generate_inconsistent_date(),
                _generate_impossible_date(),
            ]),
            "CLIENTE": _generate_bad_text(random.choice(SAMPLE_NAMES)),
            "PRODUCTO": random.choice([
                _generate_bad_text(random.choice(SAMPLE_PRODUCTS)),
                None,
                "",
                "Producto sin especificar",
            ]),
            "CANTIDAD": random.choice([
                str(random.randint(1, 50)),
                f"{random.randint(1, 50)} unidades",
                "varios",
                None,
                "N/A",
            ]),
            "PRECIO": random.choice([
                f"${random.randint(10, 1000)}",
                f"{random.randint(10, 1000)}€",
                str(random.randint(10, 1000)),
                "Gratis",
                None,
            ]),
            "NOTAS": random.choice([
                "<p>Orden urgente</p>",
                "  Entregar en recepción  ",
                "Cliente VIP®™",
                None,
                "",
                "Llamar antes de entregar\nTel: 555-0123",
            ]),
            "ESTADO": random.choice([1, 2, 3, None, "pendiente", "completado"]),
            "DIRECCION_ENVIO": random.choice([
                f"Calle {random.choice(SAMPLE_CITIES)} #{random.randint(1, 200)}",
                None,
                "",
                "Por determinar",
            ]),
            "TELF_CONTACTO": telefono,
        })

    if rows:
        rows.append(rows[0].copy())
        rows.append(rows[1].copy())

    return rows

def _generate_bad_clientes_data(num_rows: int) -> List[Dict[str, Any]]:
    rows = []
    for i in range(1, num_rows + 1):
        telefono = random.choice(SAMPLE_PHONE_FORMATS)
        if telefono is not None:
            telefono = _generate_bad_text(str(telefono))

        rows.append({
            "ID_CLIENTE": i if random.random() > 0.3 else None,
            "NOMBRE_CLIENTE": _generate_bad_text(random.choice(SAMPLE_NAMES)),
            "DIRECCION_COMPLETA": random.choice([
                f"Calle {random.choice(SAMPLE_CITIES)} #{random.randint(1, 200)}, Piso {random.randint(1, 10)}",
                f"Av. Principal {random.randint(1, 100)}, {random.choice(SAMPLE_CITIES)}",
                None,
                "Sin dirección registrada",
            ]),
            "TELEFONOS": telefono,
            "EMAILS": random.choice([
                f"cliente{i}@email.com, cliente{i}_alt@email.com",
                f"{random.choice(SAMPLE_NAMES).lower().replace(' ', '.')}@empresa.com",
                None,
                "",
            ]),
            "PROVINCIA_CIUDAD": f"{random.choice(SAMPLE_CITIES)}, {random.choice(SAMPLE_CITIES)}",
            "CODIGO_POSTAL_PAIS": f"{random.randint(10000, 99999)}-{random.choice(['España', 'México', 'Argentina', 'Colombia'])}",
            "FAX": random.choice(["555-9999", None, "", "No disponible"]),
            "DIRECCION_ALTERNATIVA": random.choice([
                None,
                "",
                f"Av. Alterna {random.randint(1, 50)}, {random.choice(SAMPLE_CITIES)}",
            ]),
            "OBSERVACIONES": random.choice([
                None,
                "",
                "<b>Cliente frecuente</b>",
                "  Sin observaciones  ",
            ]),
        })
    return rows

def _generate_bad_productos_data(num_rows: int) -> List[Dict[str, Any]]:
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
                "Consultar",
            ]),
            "CATEGORIAS": random.choice([
                f"{random.choice(SAMPLE_CATEGORIES)}, {random.choice(SAMPLE_CATEGORIES)}",
                f"{random.choice(['Tecnología', 'Accesorios', 'Periféricos'])}",
                None,
                "",
            ]),
            "PROVEEDORES_ID": random.choice([
                f"PROV_{random.randint(1, 10)}, PROV_{random.randint(11, 20)}",
                str(random.randint(1, 50)),
                None,
            ]),
            "TALLAS_DISPONIBLES": random.choice([
                "S,M,L,XL",
                "Única",
                "38,40,42,44",
                None,
                "N/A",
            ]),
            "PROVEEDOR_NOMBRE": random.choice([
                f"Proveedor {random.choice(SAMPLE_CITIES)}",
                None,
                "",
                "  ",
            ]),
            "STOCK_SEGURIDAD": random.choice([
                str(random.randint(0, 100)),
                "N/A",
                None,
                "",
                "Sin definir",
            ]),
            "UBICACION_ALMACEN": random.choice([
                f"Almacén-{random.choice(['A', 'B', 'Z', 'N/A'])}-Estante{random.randint(1, 50)}",
                None,
                "",
            ]),
        })
    return rows

def _generate_bad_todo_en_uno_data(num_rows: int) -> List[Dict[str, Any]]:
    tipos = ["CLIENTE", "PRODUCTO", "EMPLEADO", "ORDEN", "PROVEEDOR"]
    rows = []
    for i in range(1, num_rows + 1):
        tipo = random.choice(tipos)
        nombre = random.choice(SAMPLE_NAMES)
        producto = random.choice(SAMPLE_PRODUCTS)
        ciudad = random.choice(SAMPLE_CITIES)

        if tipo == "CLIENTE":
            nombre_o_desc = nombre
            cant_o_precio = random.choice(SAMPLE_PHONE_FORMATS)
            if isinstance(cant_o_precio, str) and random.random() < 0.3:
                cant_o_precio = _generate_bad_text(cant_o_precio)
            fecha_o_dir = ciudad
            estado = random.choice(["Activo", "Inactivo", "VIP", "Moroso", None])
        elif tipo == "PRODUCTO":
            nombre_o_desc = producto
            cant_o_precio = random.choice([f"${random.randint(10, 999)}", f"{random.randint(10, 999)}€", "Consultar", None])
            fecha_o_dir = f"Stock: {random.randint(0, 100)}"
            estado = random.choice(["Disponible", "Agotado", "Descatalogado", None])
        elif tipo == "EMPLEADO":
            nombre_o_desc = nombre
            cant_o_precio = random.choice([f"${random.randint(20000, 80000)}", f"{random.randint(20000, 80000)} euros", None])
            fecha_o_dir = random.choice(["Ventas", "IT", "RRHH", "Marketing", None])
            estado = random.choice(["Activo", "De baja", "Vacaciones", "Despedido", None])
        elif tipo == "ORDEN":
            nombre_o_desc = f"ORD-{random.randint(1000, 9999)}"
            cant_o_precio = random.choice([f"{random.randint(1, 50)} unid.", "varios", None])
            fecha_o_dir = _generate_inconsistent_date()
            if random.random() < 0.2:
                fecha_o_dir = _generate_impossible_date()
            estado = random.choice([1, 2, 3, "pendiente", "completado", None])
        else:
            nombre_o_desc = f"PROV-{random.choice(SAMPLE_CITIES)}"
            cant_o_precio = random.choice([f"${random.randint(1000, 99999)}", None])
            fecha_o_dir = ciudad
            estado = random.choice(["Activo", "Suspendido", None])

        if isinstance(cant_o_precio, str) and len(cant_o_precio) > 200:
            cant_o_precio = cant_o_precio[:200]

        rows.append({
            "ID": i if random.random() > 0.3 else None,
            "TIPO_REGISTRO": tipo,
            "NOMBRE_O_DESCRIPCION": _generate_bad_text(nombre_o_desc) if isinstance(nombre_o_desc, str) else nombre_o_desc,
            "CANT_O_PRECIO": str(cant_o_precio) if not isinstance(cant_o_precio, str) else cant_o_precio,
            "FECHA_O_DIRECCION": str(fecha_o_dir) if not isinstance(fecha_o_dir, str) else fecha_o_dir,
            "ESTADO_O_ACTIVO": str(estado) if estado is not None else None,
            "OBSERVACIONES": random.choice([None, "", "<b>Urgente</b>", "  Pendiente  ", "Verificado®™"]),
            "FLAG_S_N": random.choice(["S", "N", None, ""]),
            "FLAG_Y_N": random.choice(["Y", "N", None, ""]),
            "FLAG_1_0": random.choice(["1", "0", None, "", "true", "false"]),
            "FLAG_T_F": random.choice(["T", "F", None, ""]),
            "EXTRA_INFO": random.choice([None, "", f"INFO_{i}", "<xml>extra</xml>", "  "]),
            "REPETIDA_1": nombre_o_desc if isinstance(nombre_o_desc, str) else str(nombre_o_desc),
        })
    return rows

def _generate_bad_categorias_data(num_rows: int) -> List[Dict[str, Any]]:
    categorias = [
        {"nombre": "Electrónica", "padre": None, "nivel": "0"},
        {"nombre": "Hogar", "padre": None, "nivel": "0"},
        {"nombre": "Ropa", "padre": None, "nivel": "0"},
        {"nombre": "Computadoras", "padre": 1, "nivel": "1"},
        {"nombre": "Audio", "padre": 1, "nivel": "1"},
        {"nombre": "Cocina", "padre": 2, "nivel": "1"},
        {"nombre": "Muebles", "padre": 2, "nivel": "2"},
        {"nombre": "Laptops", "padre": 4, "nivel": "2"},
        {"nombre": "Auriculares", "padre": 5, "nivel": "Dos"},
        {"nombre": "Ciclo_roto_A", "padre": 11, "nivel": "3"},
        {"nombre": "Ciclo_roto_B", "padre": 10, "nivel": "tres"},
        {"nombre": "Fantasma", "padre": 99, "nivel": None},
        {"nombre": None, "padre": None, "nivel": "0"},
        {"nombre": "  ", "padre": None, "nivel": ""},
    ]

    rows = []
    for i, cat in enumerate(categorias[:num_rows], 1):
        nombre = cat.get("nombre")
        if nombre and i <= len(categorias):
            nombre = random.choice([
                _generate_bad_text(nombre),
                random.choice(SAMPLE_MULTIBYTE),
                nombre,
            ])

        rows.append({
            "ID": i,
            "NOMBRE": nombre,
            "CAT_PADRE_ID": cat["padre"],
            "NIVEL": random.choice([cat["nivel"], str(random.randint(0, 5)), None, "", "N/A"]),
            "ACTIVO": random.choice(["Y", "N", "1", "0", "S", None]),
            "RUTA_COMPLETA": random.choice([
                f"/{cat['nombre']}" if cat['nombre'] else None,
                None,
                "",
                "  /  ",
            ]) if i <= len(categorias) else random.choice([None, ""]),
        })
    return rows

def _generate_bad_transacciones_data(num_rows: int) -> List[Dict[str, Any]]:
    tipos_posibles = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "1", "2", "9", "X", "Z"]
    rows = []
    for i in range(1, num_rows + 1):
        monto = random.choice([
            f"{random.uniform(0.01, 999999.99):.2f}",
            f"{random.uniform(0.01, 999999.99):.20f}",
            f"${random.uniform(10, 1000):.2f}",
            f"{random.uniform(10, 1000):.2f} EUR",
            "Gratis",
            None,
            "-150.00",
        ])

        id_val = i
        if random.random() < 0.15 and i > 1:
            id_val = -i
        elif random.random() < 0.1:
            id_val = None

        confirmado = random.choice(["Y", "N", "S", "1", "0", None, ""])

        desc = random.choice([
            _generate_bad_text(f"Transacción {i}"),
            "<script>alert('hack')</script>",
            "  Pago  ",
            None,
            "",
        ])

        fecha = random.choice([
            _generate_impossible_date(),
            _generate_timezone_date(),
            _generate_inconsistent_date(),
        ])

        rows.append({
            "ID": id_val,
            "DESCRIPCION": desc,
            "MONTO": monto,
            "FECHA": fecha,
            "TIPO": random.choice(tipos_posibles),
            "MONEDA": random.choice(SAMPLE_CURRENCIES),
            "CONFIRMADO": confirmado,
        })
    return rows

def _generate_bad_empleados_historial_data(num_rows: int) -> List[Dict[str, Any]]:
    rows = []
    for i in range(1, num_rows + 1):
        salario_anterior = random.randint(20000, 80000)
        tipo_cambio = random.choice(["aumento", "disminución", "promoción", "cambio categoría"])

        if tipo_cambio == "aumento":
            salario_nuevo = salario_anterior - random.randint(1000, 10000)
        elif tipo_cambio == "disminución":
            salario_nuevo = salario_anterior + random.randint(1000, 10000)
        else:
            salario_nuevo = salario_anterior

        fecha_cambio = _generate_inconsistent_date()
        if random.random() < 0.2:
            fecha_cambio = _generate_impossible_date()

        rows.append({
            "ID": i if random.random() > 0.2 else None,
            "EMPLEADO_ID": random.choice([
                random.randint(1, 100),
                -1,
                9999,
                None,
            ]),
            "SALARIO_ANTERIOR": random.choice([
                str(salario_anterior),
                f"${salario_anterior}",
                None,
                "N/A",
            ]),
            "SALARIO_NUEVO": random.choice([
                str(salario_nuevo),
                f"${salario_nuevo}",
                None,
                "Sin definir",
            ]),
            "FECHA_CAMBIO": fecha_cambio,
            "TIPO_CAMBIO": random.choice([
                tipo_cambio,
                "DESCONOCIDO",
                None,
                "",
                "  ",
            ]),
            "OBSERVACIONES": random.choice([
                _generate_bad_text("Cambio registrado"),
                None,
                "",
                "<b>Revisar</b>",
            ]),
        })
    return rows

def _generate_bad_configuracion_data(num_rows: int) -> List[Dict[str, Any]]:
    configs = [
        {"clave": "MAX_INTENTOS", "valor": "5", "tipo": "NUMERO"},
        {"clave": "emailAdmin", "valor": "admin@empresa.com", "tipo": "EMAIL"},
        {"clave": "ruta_backup", "valor": "/var/backups/db/", "tipo": "TEXTO"},
        {"clave": "COLOR_TEMA", "valor": "#FF5733", "tipo": "COLOR"},
        {"clave": "tiempo_limite_sesion", "valor": "3600", "tipo": "NUMERO"},
        {"clave": "apiKeys", "valor": "sk-abc123...", "tipo": "SECRETO"},
        {"clave": "CONFIG_JSON", "valor": '{"host": "localhost", "port": 8080}', "tipo": "JSON"},
        {"clave": "XML_config", "valor": "<config><puerto>8080</puerto></config>", "tipo": "XML"},
        {"clave": "logoBase64", "valor": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==", "tipo": "BASE64"},
        {"clave": "max_conn", "valor": None, "tipo": None},
        {"clave": "  ", "valor": "", "tipo": ""},
        {"clave": None, "valor": "123", "tipo": "NUMERO"},
        {"clave": "activar_notificaciones", "valor": "SI", "tipo": "BOOLEAN"},
        {"clave": "modoDebug", "valor": "1", "tipo": "FLAG"},
        {"clave": "MAX", "valor": "999999999999999999999999999", "tipo": "NUMERO"},
    ]

    rows = []
    for i, cfg in enumerate(configs[:num_rows], 1):
        ultima_mod = random.choice([
            _generate_inconsistent_date(),
            _generate_impossible_date(),
            "nunca",
            None,
        ])

        rows.append({
            "ID": i if random.random() > 0.1 else None,
            "CLAVE": cfg["clave"],
            "VALOR": random.choice([
                cfg["valor"],
                _generate_bad_text(cfg["valor"]) if cfg["valor"] else None,
            ]) if random.random() > 0.2 else None,
            "TIPO_DATO": random.choice([cfg["tipo"], "", None, "DESCONOCIDO"]),
            "ACTIVO": random.choice(["Y", "N", "1", "0", None]),
            "ULTIMA_MODIFICACION": ultima_mod,
        })
    return rows
