import pytest

from carga_partidos import (
    agregar_partido,
    armar_partido,
    borrar_partido,
    editar_partido,
    parsear_mensaje_whatsapp,
    texto_whatsapp_equipos,
)

CONOCIDOS = ["Franco", "José MB", "Hernán", "Fer F", "Nacho F", "René"]
CSV = (
    "Fecha,Equipo1,Equipo2,Ganador\r\n"
    "5/10/2026,Franco;Hernán,José MB;René,Equipo2\r\n"
    "12/10/2026,Franco;René,José MB;Hernán,Equipo1\r\n"
)


def test_parsea_el_mensaje_que_genera_la_app():
    texto = texto_whatsapp_equipos(["Franco", "José MB"], ["Hernán", "René"])
    eq1, eq2, avisos = parsear_mensaje_whatsapp(texto, CONOCIDOS)
    assert (eq1, eq2, avisos) == (["Franco", "José MB"], ["Hernán", "René"], [])


def test_tolera_ediciones_de_whatsapp():
    texto = (
        "[19/10/26, 20:15] Fran: Equipos del lunes ⚽\n"
        "*Claras:* franco, jose mb\n"
        "Oscuras 🖤\n"
        "1. Hernan\n"
        "2. René\n"
    )
    eq1, eq2, avisos = parsear_mensaje_whatsapp(texto, CONOCIDOS)
    assert eq1 == ["Franco", "José MB"]
    assert eq2 == ["Hernán", "René"]
    assert len(avisos) == 1 and "ignoró" in avisos[0]


def test_avisa_jugador_desconocido_con_sugerencia():
    _, eq2, avisos = parsear_mensaje_whatsapp("Clara:\nFranco\nOscura:\nNacho G;Hernán", CONOCIDOS)
    assert eq2 == ["Nacho G", "Hernán"]
    assert any("Nacho F" in a for a in avisos)


def test_falta_un_equipo():
    with pytest.raises(ValueError):
        parsear_mensaje_whatsapp("Clara:\nFranco;Hernán", CONOCIDOS)


@pytest.mark.parametrize("fecha", ["14/9/20226", "31/2/2026", "2026-10-19", ""])
def test_rechaza_fechas_invalidas(fecha):
    with pytest.raises(ValueError):
        armar_partido(fecha, ["Franco"], ["Hernán"], "Equipo1")


def test_rechaza_jugador_en_ambos_equipos():
    with pytest.raises(ValueError, match="Franco"):
        armar_partido("19/10/2026", ["Franco"], ["Franco", "Hernán"], "Equipo1")


def test_agregar_editar_borrar_preserva_formato():
    partido = armar_partido("19/10/2026", ["Fer F"], ["Nacho F"], "Empate")
    nuevo = agregar_partido(CSV, partido)
    assert nuevo == CSV + "19/10/2026,Fer F,Nacho F,Empate\r\n"

    editado = editar_partido(nuevo, 3, {**partido, "Ganador": "Equipo2"})
    assert editado.endswith("19/10/2026,Fer F,Nacho F,Equipo2\r\n")
    assert editado.count("\r\n") == 4

    assert borrar_partido(nuevo, 3) == CSV
    assert "5/10/2026" not in borrar_partido(CSV, 1)
    with pytest.raises(ValueError):
        borrar_partido(CSV, 3)


def test_no_permite_cargar_dos_veces_la_misma_fecha():
    partido = armar_partido("12/10/2026", ["Fer F"], ["Nacho F"], "Equipo1")
    with pytest.raises(ValueError, match="Ya hay un partido"):
        agregar_partido(CSV, partido)
