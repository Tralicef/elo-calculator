"""Funciones puras para cargar, editar y borrar partidos del historial (data.csv)."""
from __future__ import annotations

import csv
import difflib
import io
import re
import unicodedata
from datetime import date, datetime

COLUMNAS = ["Fecha", "Equipo1", "Equipo2", "Ganador"]
GANADORES = ["Equipo1", "Equipo2", "Empate"]

# Prefijos que agrega WhatsApp al copiar mensajes, p. ej.
# "[19/10/26, 20:15] Franco: " (iOS) o "19/10/26 20:15 - Franco: " (Android).
_PREFIJO_WHATSAPP = re.compile(
    r"^\[?\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:[ap]\.?\s?m\.?)?\]?\s*-?\s*[^:]+:\s*",
    re.IGNORECASE,
)
_ENCABEZADO = re.compile(r"^(clar[oa]s?|oscur[oa]s?|equipo\s*([12]))\b")
_MARCADOR_LISTA = re.compile(r"^(\d+[.)]|[-•*·])\s*")


def texto_whatsapp_equipos(eq1: list[str], eq2: list[str]) -> str:
    """Devuelve equipos en formato fácil de copiar a WhatsApp."""
    claras = ";".join(eq1)
    oscuras = ";".join(eq2)
    return f"Clara:\n{claras}\n\nOscura:\n{oscuras}"


def _normalizar(texto: str) -> str:
    """Minúsculas, sin acentos y con espacios colapsados (para comparar nombres)."""
    sin_acentos = unicodedata.normalize("NFKD", texto)
    sin_acentos = "".join(c for c in sin_acentos if not unicodedata.combining(c))
    return " ".join(sin_acentos.lower().split())


def _limpiar_nombre(nombre: str) -> str:
    """Saca emojis, viñetas y signos sueltos alrededor de un nombre."""
    nombre = _MARCADOR_LISTA.sub("", nombre.strip())
    nombre = "".join(c for c in nombre if c.isalnum() or c in " .'-")
    return " ".join(nombre.split()).strip(" .-")


def parsear_mensaje_whatsapp(
    texto: str, jugadores_conocidos: list[str] | None = None
) -> tuple[list[str], list[str], list[str]]:
    """
    Extrae los equipos del mensaje "Clara: ... / Oscura: ..." pegado desde WhatsApp.

    Acepta nombres separados por ";", "," o uno por línea, y encabezados
    "Clara"/"Oscura" (o "Equipo 1"/"Equipo 2"). Los nombres se mapean a los
    jugadores conocidos ignorando mayúsculas y acentos.

    Args:
        texto: Mensaje copiado de WhatsApp
        jugadores_conocidos: Jugadores que ya figuran en el historial

    Returns:
        tuple: (equipo1, equipo2, avisos) donde equipo1 es Clara y equipo2 Oscura

    Raises:
        ValueError: Si no se encuentran los dos equipos
    """
    conocidos = {_normalizar(j): j for j in (jugadores_conocidos or [])}
    equipos: dict[int, list[str]] = {1: [], 2: []}
    avisos: list[str] = []
    actual = None

    for linea in texto.splitlines():
        linea = _PREFIJO_WHATSAPP.sub("", linea.strip()).strip(" *_~")
        if not linea:
            continue

        encabezado = _ENCABEZADO.match(_normalizar(_limpiar_nombre(linea)))
        if encabezado:
            es_oscura = encabezado.group(1).startswith("oscur") or encabezado.group(2) == "2"
            actual = 2 if es_oscura else 1
            linea = linea.split(":", 1)[1] if ":" in linea else ""

        if actual is None:
            avisos.append(f"Se ignoró la línea «{linea}» (está antes de «Clara:»).")
            continue

        for crudo in re.split(r"[;,]", linea):
            nombre = _limpiar_nombre(crudo)
            if not nombre:
                continue
            clave = _normalizar(nombre)
            if clave in conocidos:
                nombre = conocidos[clave]
            elif conocidos:
                parecido = difflib.get_close_matches(clave, conocidos.keys(), n=1, cutoff=0.75)
                sugerencia = f" ¿Quisiste decir «{conocidos[parecido[0]]}»?" if parecido else ""
                avisos.append(f"«{nombre}» no jugó nunca: se carga como jugador nuevo.{sugerencia}")
            if nombre in equipos[actual]:
                avisos.append(f"«{nombre}» estaba repetido en el mismo equipo; se cuenta una vez.")
                continue
            equipos[actual].append(nombre)

    if not equipos[1] or not equipos[2]:
        raise ValueError(
            "No encontré los dos equipos. El mensaje tiene que tener «Clara:» y «Oscura:»."
        )
    if len(equipos[1]) != len(equipos[2]):
        avisos.append(
            f"Los equipos tienen distinta cantidad: Clara {len(equipos[1])}, Oscura {len(equipos[2])}."
        )
    return equipos[1], equipos[2], avisos


def formatear_fecha(dia: date) -> str:
    """Formatea la fecha como en data.csv (d/m/aaaa)."""
    return f"{dia.day}/{dia.month}/{dia.year}"


def armar_partido(fecha: str, equipo1: list[str], equipo2: list[str], ganador: str) -> dict:
    """
    Valida y arma la fila de un partido.

    Raises:
        ValueError: Si la fecha, los equipos o el ganador no son válidos
    """
    try:
        dia = datetime.strptime(fecha.strip(), "%d/%m/%Y")
    except ValueError:
        raise ValueError(f"Fecha inválida «{fecha}»: usá d/m/aaaa.") from None
    if not 2020 <= dia.year <= 2100:
        raise ValueError(f"Año fuera de rango en «{fecha}».")
    if not equipo1 or not equipo2:
        raise ValueError("Faltan jugadores en algún equipo.")
    repetidos = sorted(set(equipo1) & set(equipo2))
    if repetidos:
        raise ValueError(f"Jugadores en ambos equipos: {', '.join(repetidos)}.")
    if ganador not in GANADORES:
        raise ValueError(f"Ganador inválido «{ganador}».")
    return {
        "Fecha": fecha.strip(),
        "Equipo1": ";".join(equipo1),
        "Equipo2": ";".join(equipo2),
        "Ganador": ganador,
    }


def _leer_filas(texto_csv: str) -> tuple[list[dict], str]:
    """Devuelve las filas del CSV y el fin de línea que usa el archivo."""
    terminador = "\r\n" if "\r\n" in texto_csv else "\n"
    return list(csv.DictReader(io.StringIO(texto_csv))), terminador


def _escribir_filas(filas: list[dict], terminador: str) -> str:
    salida = io.StringIO()
    writer = csv.DictWriter(salida, fieldnames=COLUMNAS, lineterminator=terminador)
    writer.writeheader()
    writer.writerows({col: fila.get(col, "") for col in COLUMNAS} for fila in filas)
    return salida.getvalue()


def _validar_numero(filas: list[dict], numero: int) -> None:
    if not 1 <= numero <= len(filas):
        raise ValueError(f"No existe el partido #{numero}.")


def agregar_partido(texto_csv: str, partido: dict) -> str:
    """
    Agrega un partido al final del CSV y devuelve el texto nuevo.

    Raises:
        ValueError: Si ya hay un partido cargado con esa fecha (evita cargas dobles)
    """
    filas, terminador = _leer_filas(texto_csv)
    if any(fila["Fecha"].strip() == partido["Fecha"] for fila in filas):
        raise ValueError(
            f"Ya hay un partido cargado el {partido['Fecha']}. "
            "Si está mal, corregilo en «Editar / borrar»."
        )
    filas.append(partido)
    return _escribir_filas(filas, terminador)


def editar_partido(texto_csv: str, numero: int, partido: dict) -> str:
    """Reemplaza el partido #numero (1 = primero del historial)."""
    filas, terminador = _leer_filas(texto_csv)
    _validar_numero(filas, numero)
    filas[numero - 1] = partido
    return _escribir_filas(filas, terminador)


def borrar_partido(texto_csv: str, numero: int) -> str:
    """Borra el partido #numero (1 = primero del historial)."""
    filas, terminador = _leer_filas(texto_csv)
    _validar_numero(filas, numero)
    del filas[numero - 1]
    return _escribir_filas(filas, terminador)
