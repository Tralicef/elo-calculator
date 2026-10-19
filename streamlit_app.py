from __future__ import annotations

import hmac
import io
import json
from collections import defaultdict
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from trueskill import Rating

from carga_partidos import (
    COLUMNAS,
    GANADORES,
    agregar_partido,
    armar_partido,
    borrar_partido,
    editar_partido,
    formatear_fecha,
    parsear_mensaje_whatsapp,
    texto_whatsapp_equipos,
)
from elo_calculator import (
    actualizar_elo,
    actualizar_elo_desde_dataframe,
    calcular_evolucion_elo_desde_dataframe,
    crear_equipos_balanceados,
    mostrar_ranking,
    mostrar_ranking_desde_dataframe,
    parse_team,
)
from matriz_jugadores import (
    crear_matriz_ponderada_correcta,
    procesar_datos_partidos,
)
from repositorio_github import ConflictoError, guardar_archivo, leer_archivo


IMAGE_FILES = [
    "matriz_jugadores.jpg",
    "matriz_ponderada.jpg",
    "red_conexiones.jpg",
    "top_duplas.jpg",
    "peores_duplas.jpg",
    "duplas_rendimiento_top.jpg",
    "duplas_rendimiento_peores.jpg",
    "trios_rendimiento_top.jpg",
    "trios_rendimiento_peores.jpg",
    "cuartetos_rendimiento_top.jpg",
    "cuartetos_rendimiento_peores.jpg",
]


ETIQUETAS_GANADOR = {
    "Equipo1": "Clara (Equipo 1)",
    "Equipo2": "Oscura (Equipo 2)",
    "Empate": "Empate",
}
GANADOR_POR_ETIQUETA = {etiqueta: ganador for ganador, etiqueta in ETIQUETAS_GANADOR.items()}


def leer_secret(clave: str):
    """Lee un valor de st.secrets sin mostrar errores si no existe secrets.toml."""
    try:
        if not st.secrets.load_if_toml_exists():
            return None
        return st.secrets.get(clave)
    except Exception:
        return None


def config_github() -> dict | None:
    """Config de GitHub desde st.secrets; None si no está (se usa el data.csv local)."""
    cfg = leer_secret("github")
    if not cfg or not cfg.get("token") or not cfg.get("repo"):
        return None
    return {
        "token": cfg["token"],
        "repo": cfg["repo"],
        "rama": cfg.get("branch", "main"),
        "ruta": cfg.get("path", "data.csv"),
    }


@st.cache_data(ttl=60, show_spinner=False)
def _leer_github(token: str, repo: str, ruta: str, rama: str) -> tuple[str, str]:
    return leer_archivo(token, repo, ruta, rama)


def leer_historial(data_path: Path) -> tuple[str, str | None]:
    """Devuelve (contenido del CSV, sha en GitHub o None si es local)."""
    cfg = config_github()
    if cfg:
        return _leer_github(cfg["token"], cfg["repo"], cfg["ruta"], cfg["rama"])
    if data_path.exists():
        return data_path.read_bytes().decode("utf-8"), None
    st.info(f"No se encontró {data_path}. Cargá partidos en la pestaña 'Cargar partido'.")
    return ",".join(COLUMNAS) + "\n", None


def guardar_historial(data_path: Path, contenido: str, sha: str | None, mensaje: str) -> None:
    """Guarda el CSV: commit en GitHub si está configurado, si no en disco."""
    cfg = config_github()
    if cfg:
        guardar_archivo(
            cfg["token"], cfg["repo"], cfg["ruta"], cfg["rama"], contenido, sha, mensaje
        )
        _leer_github.clear()
    else:
        data_path.write_bytes(contenido.encode("utf-8"))


def password_ok(ingresada: str) -> bool:
    """Valida la contraseña de carga (si no hay `admin_password` en secrets, no se pide)."""
    esperada = leer_secret("admin_password")
    if not esperada:
        return True
    return hmac.compare_digest(ingresada.encode("utf-8"), str(esperada).encode("utf-8"))


def calcular_rendimiento_grupos(df: pd.DataFrame, size: int, min_partidos: int) -> pd.DataFrame:
    """Calcula total y winrate por grupo de tamaño size."""
    records: dict[tuple[str, ...], dict[str, int]] = defaultdict(lambda: {"total": 0, "wins": 0})
    if df.empty:
        return pd.DataFrame(columns=["Grupo", "Partidos", "Victorias", "Winrate"])

    for _, row in df.iterrows():
        t1 = parse_team(row["Equipo1"])
        t2 = parse_team(row["Equipo2"])
        ganador = row.get("Ganador", "")

        for combo in combinations(t1, size):
            key = tuple(sorted(combo))
            records[key]["total"] += 1
            if ganador == "Equipo1":
                records[key]["wins"] += 1
        for combo in combinations(t2, size):
            key = tuple(sorted(combo))
            records[key]["total"] += 1
            if ganador == "Equipo2":
                records[key]["wins"] += 1

    data = []
    for combo, vals in records.items():
        if vals["total"] >= min_partidos:
            winrate = vals["wins"] / vals["total"] if vals["total"] else 0
            data.append(
                {"Grupo": "; ".join(combo), "Partidos": vals["total"], "Victorias": vals["wins"], "Winrate": winrate}
            )

    if not data:
        return pd.DataFrame(columns=["Grupo", "Partidos", "Victorias", "Winrate"])

    df_stats = pd.DataFrame(data)
    df_stats["Winrate"] = (df_stats["Winrate"] * 100).round(1)
    return df_stats.sort_values(["Winrate", "Partidos"], ascending=[False, False])


def mostrar_top_bottom(df_stats: pd.DataFrame, title: str, n: int = 10) -> None:
    """Muestra mejores y peores grupos como barras."""
    top = df_stats.head(n)
    bottom = df_stats.sort_values(["Winrate", "Partidos"], ascending=[True, False]).head(n)

    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"Mejores {title.lower()}")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=top,
            y="Grupo",
            x="Winrate",
            ax=ax,
            palette="Greens",
        )
        ax.set_xlim(0, 100)
        ax.set_xlabel("Winrate %")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.dataframe(top.reset_index(drop=True), use_container_width=True)
    with col2:
        st.caption(f"Peores {title.lower()}")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=bottom,
            y="Grupo",
            x="Winrate",
            ax=ax,
            palette="Reds",
        )
        ax.set_xlim(0, 100)
        ax.set_xlabel("Winrate %")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.dataframe(bottom.reset_index(drop=True), use_container_width=True)


def calcular_stats_recientes(df: pd.DataFrame, ultimos_n: int) -> pd.DataFrame:
    """Calcula partidos, victorias y winrate en los últimos N partidos."""
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"partidos": 0, "victorias": 0})
    if df.empty:
        return pd.DataFrame(columns=["Jugador", "Reciente"])

    for _, row in df.tail(ultimos_n).iterrows():
        equipo1 = parse_team(row["Equipo1"])
        equipo2 = parse_team(row["Equipo2"])
        for jugador in equipo1 + equipo2:
            stats[jugador]["partidos"] += 1

        if row["Ganador"] == "Equipo1":
            for jugador in equipo1:
                stats[jugador]["victorias"] += 1
        elif row["Ganador"] == "Equipo2":
            for jugador in equipo2:
                stats[jugador]["victorias"] += 1

    records = []
    for jugador, vals in stats.items():
        partidos = vals["partidos"]
        victorias = vals["victorias"]
        winrate = (victorias / partidos * 100) if partidos else 0
        records.append(
            {
                "Jugador": jugador,
                "Reciente": f"{victorias}-{partidos - victorias} ({winrate:.1f}%)",
            }
        )
    return pd.DataFrame(records)


def calcular_df_ranking(
    df: pd.DataFrame,
    data_path: Path | None = None,
    min_partidos: int = 5,
    ultimos_n: int = 10,
) -> pd.DataFrame:
    """Devuelve el ranking listo para mostrar en Streamlit."""
    if df.empty:
        return pd.DataFrame()

    if data_path is not None:
        ratings = actualizar_elo(str(data_path))
        ranking = mostrar_ranking(ratings, str(data_path), min_partidos=min_partidos)
    else:
        ratings = actualizar_elo_desde_dataframe(df)
        ranking = mostrar_ranking_desde_dataframe(ratings, df, min_partidos=min_partidos)

    df_rank = pd.DataFrame(
        ranking, columns=["Jugador", "Mu", "Sigma", "Partidos", "Victorias"]
    )
    if df_rank.empty:
        return df_rank

    if len(df) > 1:
        df_previo = df.iloc[:-1].copy()
        ratings_previos = actualizar_elo_desde_dataframe(df_previo)
        ranking_previo = mostrar_ranking_desde_dataframe(
            ratings_previos, df_previo, min_partidos=min_partidos
        )
        posiciones_previas = {
            jugador: posicion
            for posicion, (jugador, *_resto) in enumerate(ranking_previo, start=1)
        }
        cambios = []
        for posicion_actual, jugador in enumerate(df_rank["Jugador"], start=1):
            posicion_previa = posiciones_previas.get(jugador)
            cambio = (
                None
                if posicion_previa is None
                else posicion_previa - posicion_actual
            )
            cambios.append(_formatear_cambio_posicion(cambio))
        df_rank["Cambio"] = cambios
    else:
        df_rank["Cambio"] = "Nuevo"

    df_rank["Derrotas"] = df_rank["Partidos"] - df_rank["Victorias"]
    df_rank["Récord"] = (
        df_rank["Victorias"].astype(str) + "-" + df_rank["Derrotas"].astype(str)
    )
    df_rank["Win %"] = (df_rank["Victorias"] / df_rank["Partidos"].clip(lower=1)) * 100
    df_rank["Win %"] = df_rank["Win %"].round(1)

    recientes = calcular_stats_recientes(df, ultimos_n)
    if not recientes.empty:
        df_rank = df_rank.merge(recientes, on="Jugador", how="left")
    else:
        df_rank["Reciente"] = "-"
    df_rank["Reciente"] = df_rank["Reciente"].fillna("-")

    return df_rank[
        ["Jugador", "Cambio", "Mu", "Sigma", "Récord", "Partidos", "Victorias", "Win %", "Reciente"]
    ]


def mu_promedio_equipo(equipo: list[str], ratings: dict) -> float:
    """Calcula el Mu promedio de un equipo."""
    if not equipo:
        return 0.0
    return sum(ratings.get(jugador, Rating()).mu for jugador in equipo) / len(equipo)


def detalle_equipo(equipo: list[str], ratings: dict) -> pd.DataFrame:
    """Devuelve jugadores de un equipo con rating para mostrar en tablas."""
    return pd.DataFrame(
        [
            {
                "Jugador": jugador,
                "Mu": round(ratings.get(jugador, Rating()).mu, 1),
                "Sigma": round(ratings.get(jugador, Rating()).sigma, 1),
            }
            for jugador in equipo
        ]
    )


def render_resumen_ultimo_partido(df: pd.DataFrame) -> None:
    """Muestra un resumen compacto del último partido cargado."""
    if df.empty:
        return

    ultimo = df.iloc[-1]
    numero_partido = len(df)
    equipo1 = parse_team(ultimo["Equipo1"])
    equipo2 = parse_team(ultimo["Equipo2"])

    st.subheader("Último partido")
    cols = st.columns(4)
    cols[0].metric("Partido", f"#{numero_partido}")
    cols[1].metric("Fecha", str(ultimo["Fecha"]))
    cols[2].metric("Resultado", str(ultimo["Ganador"]))
    cols[3].metric("Jugadores", len(set(equipo1 + equipo2)))

    ratings_despues = actualizar_elo_desde_dataframe(df)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Equipo 1 - Mu promedio", f"{mu_promedio_equipo(equipo1, ratings_despues):.1f}")
        st.dataframe(
            detalle_equipo(equipo1, ratings_despues).style.format(
                {"Mu": "{:.1f}", "Sigma": "{:.1f}"}
            ),
            use_container_width=True,
        )
    with col2:
        st.metric("Equipo 2 - Mu promedio", f"{mu_promedio_equipo(equipo2, ratings_despues):.1f}")
        st.dataframe(
            detalle_equipo(equipo2, ratings_despues).style.format(
                {"Mu": "{:.1f}", "Sigma": "{:.1f}"}
            ),
            use_container_width=True,
        )


def render_stats_jugador(df: pd.DataFrame, data_path: Path) -> None:
    st.subheader("Estadísticas por jugador")
    if df.empty:
        st.info("Sube data para ver estadísticas individuales.")
        return

    jugadores = obtener_jugadores(df)
    if not jugadores:
        st.info("No hay jugadores registrados en data.csv aún.")
        return

    default_j = obtener_jugadores_ultimo_partido(df)
    default_idx = jugadores.index(default_j[0]) if default_j and default_j[0] in jugadores else 0
    jugador_sel = st.selectbox("Jugador", options=jugadores, index=default_idx)

    colp1, colp2 = st.columns(2)
    with colp1:
        min_pares = st.slider("Mínimo partidos del par", 1, 10, 2)
    with colp2:
        min_grupo = st.slider("Mínimo partidos del grupo", 1, 15, 2)

    st.markdown("#### Con quién jugaste más / menos (matriz ponderada)")
    pares = obtener_pares_por_jugador(df, jugador_sel, min_pares)
    if not pares:
        st.info("No hay pares con suficiente muestra.")
    else:
        df_pares = pd.DataFrame(pares).sort_values(
            ["Juntos %", "Partidos"], ascending=[False, False]
        )
        st.caption("Top afinidad")
        st.dataframe(df_pares.head(5).reset_index(drop=True), use_container_width=True)
        st.caption("Menor afinidad")
        st.dataframe(
            df_pares.sort_values(["Juntos %", "Partidos"], ascending=[True, False])
            .head(5)
            .reset_index(drop=True),
            use_container_width=True,
        )

    st.markdown("#### Rendimientos con este jugador")
    for size, label in [(2, "Duplas"), (3, "Tríos"), (4, "Cuartetos")]:
        stats = calcular_rendimiento_grupos(df, size, min_grupo)
        if stats.empty:
            st.info(f"Sin {label.lower()} con suficiente muestra.")
            continue
        filtered = stats[
            stats["Grupo"].apply(
                lambda g: jugador_sel in [p.strip() for p in g.split(";")]
            )
        ]
        if filtered.empty:
            st.info(f"{jugador_sel} no tiene {label.lower()} con el mínimo seleccionado.")
            continue
        st.markdown(f"**{label} de {jugador_sel}**")
        top = filtered.head(5)
        bottom = filtered.sort_values(["Winrate", "Partidos"], ascending=[True, False]).head(5)
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Mejores")
            st.dataframe(top.reset_index(drop=True), use_container_width=True)
        with col2:
            st.caption("Peores")
            st.dataframe(bottom.reset_index(drop=True), use_container_width=True)


def parse_restricciones(text: str) -> List[List[str]]:
    grupos: List[List[str]] = []
    for line in text.strip().splitlines():
        nombres = [n.strip() for n in line.replace(";", ",").split(",") if n.strip()]
        if nombres:
            grupos.append(nombres)
    return grupos


def render_historial(df: pd.DataFrame) -> None:
    st.subheader("Historial de partidos")
    if df.empty:
        st.info("Sin datos aún.")
        return

    jugador = st.text_input("Filtrar por jugador (opcional)")
    filtered = df.copy()
    filtered["Partido"] = filtered.index + 1
    filtered = filtered[
        ["Partido"] + [col for col in filtered.columns if col != "Partido"]
    ]
    if jugador:
        filtered = filtered[
            filtered["Equipo1"].str.contains(jugador, case=False)
            | filtered["Equipo2"].str.contains(jugador, case=False)
        ]
    st.dataframe(
        filtered.sort_values("Partido", ascending=False),
        use_container_width=True,
    )
    st.caption(f"Total partidos: {len(filtered)}")


def _formatear_cambio_posicion(cambio: int | None) -> str:
    """Devuelve el texto visible para el cambio de ranking."""
    if cambio is None:
        return "Nuevo"
    if cambio > 0:
        return f"▲ {cambio}"
    if cambio < 0:
        return f"▼ {abs(cambio)}"
    return "→ 0"


def _color_cambio_posicion(valor: str) -> str:
    """Colorea la columna de cambio según subió o bajó posiciones."""
    if valor.startswith("▲"):
        return "color: #148a3b; font-weight: 700"
    if valor.startswith("▼"):
        return "color: #c7362f; font-weight: 700"
    return "color: #6b7280"


def render_ranking(df: pd.DataFrame, min_partidos: int, data_path: Path) -> None:
    st.subheader("Ranking TrueSkill")
    if df.empty:
        st.info("Sube data para calcular el ranking.")
        return

    col1, col2 = st.columns(2)
    with col1:
        min_partidos_ui = st.slider(
            "Mínimo de partidos",
            1,
            max(1, len(df)),
            min(min_partidos, max(1, len(df))),
        )
    with col2:
        max_recientes = max(1, min(20, len(df)))
        ultimos_n = st.slider(
            "Forma reciente",
            1,
            max_recientes,
            min(10, max_recientes),
        )

    df_rank = calcular_df_ranking(
        df,
        min_partidos=min_partidos_ui,
        ultimos_n=ultimos_n,
    )
    if df_rank.empty:
        st.info("No hay jugadores con los criterios elegidos.")
        return

    df_rank.index = range(1, len(df_rank) + 1)
    df_rank.index.name = "#"
    st.dataframe(
        df_rank.style.map(_color_cambio_posicion, subset=["Cambio"]).format(
            {
                "Mu": "{:.1f}",
                "Sigma": "{:.1f}",
                "Win %": "{:.1f}",
            }
        ),
        use_container_width=True,
    )

    st.markdown("#### Evolución")
    jugadores_default = df_rank["Jugador"].head(5).tolist()
    seleccion = st.multiselect(
        "Jugadores",
        options=obtener_jugadores(df),
        default=jugadores_default,
        key="ranking_evolucion_jugadores",
    )
    if not seleccion:
        st.info("Seleccioná al menos un jugador para ver la evolución.")
        return

    evolucion = calcular_evolucion_elo_desde_dataframe(df)
    evolucion = evolucion[evolucion["Jugador"].isin(seleccion)]
    if evolucion.empty:
        st.info("No hay datos de evolución para los jugadores seleccionados.")
        return
    st.line_chart(
        evolucion.pivot_table(index="Partido", columns="Jugador", values="Mu"),
        use_container_width=True,
    )


def obtener_jugadores(df: pd.DataFrame) -> list[str]:
    """Devuelve lista alfabética de jugadores que ya figuraron en data."""
    jugadores = set()
    for _, row in df.iterrows():
        for col in ("Equipo1", "Equipo2"):
            jugadores.update(parse_team(row[col]))
    return sorted(jugadores)


def obtener_jugadores_ultimo_partido(df: pd.DataFrame) -> list[str]:
    """Jugadores que aparecieron en el último registro de data.csv."""
    if df.empty:
        return []
    last = df.iloc[-1]
    return sorted(set(parse_team(last["Equipo1"]) + parse_team(last["Equipo2"])))


def obtener_pares_por_jugador(df: pd.DataFrame, jugador: str, min_partidos: int) -> list[dict]:
    """Devuelve pares con porcentaje y muestras para un jugador."""
    if df.empty or jugador not in obtener_jugadores(df):
        return []

    partidos_por_jugador = defaultdict(int)
    combinaciones_partidos = []
    for _, row in df.iterrows():
        e1 = parse_team(row["Equipo1"])
        e2 = parse_team(row["Equipo2"])
        for j in e1 + e2:
            partidos_por_jugador[j] += 1
        combinaciones_partidos.extend([e1, e2])

    jugadores_lista = sorted(partidos_por_jugador.keys())
    if jugador not in jugadores_lista:
        return []

    matriz_texto, matriz_num = crear_matriz_ponderada_correcta(
        partidos_por_jugador, combinaciones_partidos, jugadores_lista
    )
    idx = jugadores_lista.index(jugador)
    pares = []
    for j, otro in enumerate(jugadores_lista):
        if j == idx:
            continue
        frac = matriz_texto[idx][j]
        if not frac:
            continue
        try:
            juntos, total = map(int, frac.split("/"))
        except ValueError:
            continue
        if total < min_partidos:
            continue
        porcentaje = round(matriz_num[idx][j], 1) if matriz_num[idx][j] == matriz_num[idx][j] else 0.0
        pares.append({"Con": otro, "Juntos/partidos": frac, "Partidos": total, "Juntos %": porcentaje})
    return pares


def generar_sugerencias_parejas(
    jugadores: list[str], df: pd.DataFrame, min_partidos_par: int = 4
) -> tuple[list[dict], list[dict]]:
    """Devuelve listas de pares con mucha o poca historia juntos."""
    if df.empty:
        return [], []

    partidos_por_jugador = defaultdict(int)
    combinaciones_partidos = []
    for _, row in df.iterrows():
        e1 = parse_team(row["Equipo1"])
        e2 = parse_team(row["Equipo2"])
        for j in e1 + e2:
            partidos_por_jugador[j] += 1
        combinaciones_partidos.extend([e1, e2])
    en_data = [j for j in jugadores if partidos_por_jugador.get(j, 0) > 0]
    if len(en_data) < 2:
        return [], []

    matriz_texto, matriz_num = crear_matriz_ponderada_correcta(
        partidos_por_jugador, combinaciones_partidos, en_data
    )

    pares = []
    for i, j1 in enumerate(en_data):
        for j in range(i + 1, len(en_data)):
            j2 = en_data[j]
            frac = matriz_texto[i][j]
            if not frac:
                continue
            try:
                juntos, total = map(int, frac.split("/"))
            except ValueError:
                continue
            if total < min_partidos_par:
                continue
            porcentaje = matriz_num[i][j]
            pares.append(
                {"pair": (j1, j2), "frac": frac, "porcentaje": round(porcentaje, 1), "total": total}
            )

    # Alta afinidad: 70%+
    altos = sorted(
        [p for p in pares if p["porcentaje"] >= 70], key=lambda x: x["porcentaje"], reverse=True
    )[:3]
    # Baja afinidad: <=30%
    bajos = sorted(
        [p for p in pares if p["porcentaje"] <= 30], key=lambda x: x["porcentaje"]
    )[:3]
    return altos, bajos


def render_equipos(df: pd.DataFrame, data_path: Path) -> None:
    st.subheader("Armar equipos parejos")
    if df.empty:
        st.info("No hay partidos cargados. Puedes agregar jugadores manualmente y probar igualmente.")

    base_jugadores = obtener_jugadores(df)
    nuevos_text = st.text_input("Agregar jugadores nuevos (separa con coma o punto y coma)")
    jugadores_nuevos = [j.strip() for j in nuevos_text.replace(";", ",").split(",") if j.strip()]

    opciones = sorted(set(base_jugadores + jugadores_nuevos))
    default_sel = [j for j in obtener_jugadores_ultimo_partido(df) if j in opciones] or opciones
    seleccion = st.multiselect(
        "Jugadores disponibles",
        options=opciones,
        default=default_sel,
    )

    st.markdown("Restricciones (una línea por grupo, separa nombres con coma o punto y coma):")
    mismo_text = st.text_area(
        "Mismo equipo",
        value="",
        height=100,
    )
    distinto_text = st.text_area(
        "Distinto equipo",
        value="",
        height=100,
    )

    if st.button("Sugerir equipos", type="primary"):
        if len(seleccion) < 4:
            st.error("Necesitas al menos 4 jugadores.")
            return
        if len(seleccion) % 2 != 0:
            st.warning("Cantidad impar de jugadores: un equipo tendrá uno más.")
        jugadores_por_equipo = len(seleccion) // 2
        ratings = actualizar_elo_desde_dataframe(df)
        try:
            eq1, eq2, calidad = crear_equipos_balanceados(
                seleccion,
                ratings,
                jugadores_por_equipo=int(jugadores_por_equipo),
                restricciones_mismo_equipo=parse_restricciones(mismo_text),
                restricciones_distinto_equipo=parse_restricciones(distinto_text),
            )
            st.success(f"Calidad del partido: {calidad:.2%}")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Equipo 1 - Mu promedio", f"{mu_promedio_equipo(eq1, ratings):.1f}")
                st.dataframe(
                    detalle_equipo(eq1, ratings).style.format(
                        {"Mu": "{:.1f}", "Sigma": "{:.1f}"}
                    ),
                    use_container_width=True,
                )
            with col2:
                st.metric("Equipo 2 - Mu promedio", f"{mu_promedio_equipo(eq2, ratings):.1f}")
                st.dataframe(
                    detalle_equipo(eq2, ratings).style.format(
                        {"Mu": "{:.1f}", "Sigma": "{:.1f}"}
                    ),
                    use_container_width=True,
                )

            st.markdown("#### Copiar para WhatsApp")
            st.code(texto_whatsapp_equipos(eq1, eq2), language="text")

            # Sugerencias de afinidad
            altos, bajos = generar_sugerencias_parejas(seleccion, df)
            if altos or bajos:
                st.markdown("### Sugerencias según historial (matriz ponderada)")
            if altos:
                for sug in altos:
                    st.warning(
                        f"{sug['pair'][0]} y {sug['pair'][1]} jugaron juntos el {sug['porcentaje']}% de las veces ({sug['frac']}). "
                        "Probá separarlos."
                    )
            if bajos:
                for sug in bajos:
                    st.info(
                        f"{sug['pair'][0]} y {sug['pair'][1]} casi no jugaron juntos ({sug['porcentaje']}%, {sug['frac']}). "
                        "Podrías ponerlos en el mismo equipo."
                    )
        except Exception as exc:
            st.error(f"No se pudo generar equipos: {exc}")


def render_graficos() -> None:
    st.subheader("Gráficos y matrices")
    st.caption("Se generan en vivo a partir de data.csv")
    data_path = st.session_state.get("data_path", Path("data.csv"))
    df = pd.read_csv(data_path) if data_path.exists() else pd.DataFrame(columns=COLUMNAS)

    if df.empty:
        st.info("Sube data para ver los gráficos.")
        return

    min_partidos = st.slider("Mínimo de partidos para mostrar en matriz ponderada", 1, 10, 10)

    # Matriz ponderada: veces juntos / veces mismo partido
    with st.spinner("Calculando matriz ponderada..."):
        try:
            partidos_por_jugador, combinaciones_partidos = procesar_datos_partidos(str(data_path))
            st.caption(
                f"Partidos: {len(df)}, jugadores con datos: {len(partidos_por_jugador)}, combinaciones: {len(combinaciones_partidos)}"
            )
            jugadores_filtrados = [
                j for j, cnt in partidos_por_jugador.items() if cnt >= min_partidos
            ]
            st.caption(f"Jugadores en matriz (min {min_partidos}): {len(jugadores_filtrados)}")
            if not jugadores_filtrados:
                st.info("No hay suficientes partidos para generar la matriz ponderada.")
            else:
                matriz_texto, matriz_num = crear_matriz_ponderada_correcta(
                    partidos_por_jugador, combinaciones_partidos, jugadores_filtrados
                )
                df_heat = pd.DataFrame(matriz_num, index=jugadores_filtrados, columns=jugadores_filtrados)
                fig, ax = plt.subplots(figsize=(10, 7))
                sns.heatmap(
                    df_heat,
                    annot=matriz_texto,
                    fmt="",
                    cmap="Blues",
                    cbar_kws={"label": "% veces en mismo equipo"},
                    ax=ax,
                    linewidths=0.5,
                    linecolor="white",
                )
                ax.set_title("Matriz ponderada (juntos / mismo partido)")
                plt.xticks(rotation=45, ha="right")
                plt.yticks(rotation=0)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.success("Matriz ponderada generada.")
        except Exception as exc:
            st.error(f"No se pudo generar la matriz ponderada: {exc}")

    st.markdown("---")
    st.subheader("Rendimiento de grupos")
    min_grupo = st.slider("Mínimo de partidos del grupo", 1, 15, 2)

    for size, label in [(2, "Duplas"), (3, "Tríos"), (4, "Cuartetos")]:
        st.markdown(f"**{label}**")
        with st.spinner(f"Calculando {label.lower()}..."):
            try:
                stats = calcular_rendimiento_grupos(df, size, min_grupo)
                if stats.empty:
                    st.info(f"Sin {label.lower()} con al menos {min_grupo} partidos.")
                    continue
                mostrar_top_bottom(stats, f"{label} por winrate")
            except Exception as exc:
                st.error(f"No se pudieron calcular {label.lower()}: {exc}")


def ultimo_lunes() -> date:
    """Fecha por defecto para cargar: el lunes más reciente (hoy si es lunes)."""
    hoy = date.today()
    return hoy - timedelta(days=hoy.weekday())


def aplicar_cambio(
    data_path: Path,
    contenido: str,
    sha: str | None,
    clave: str,
    mensaje_commit: str,
    aviso: str,
) -> None:
    """Valida la contraseña, guarda el CSV y recarga la app mostrando `aviso`."""
    if not password_ok(clave):
        st.error("Contraseña incorrecta.")
        return
    try:
        guardar_historial(data_path, contenido, sha, mensaje_commit)
    except ConflictoError as exc:
        st.error(str(exc))
        return
    except requests.RequestException as exc:
        st.error(f"No se pudo guardar en GitHub: {exc}")
        return
    st.session_state["aviso_guardado"] = aviso
    st.rerun()


def form_resultado(
    clave_form: str,
    equipo1: list[str],
    equipo2: list[str],
    texto_csv: str,
    sha: str | None,
    data_path: Path,
) -> None:
    """Pide fecha, ganador y contraseña, y agrega el partido al historial."""
    with st.form(f"form-{clave_form}"):
        dia = st.date_input(
            "Fecha", value=ultimo_lunes(), format="DD/MM/YYYY", key=f"fecha-{clave_form}"
        )
        ganador = st.radio(
            "¿Quién ganó?",
            list(GANADOR_POR_ETIQUETA),
            index=None,
            horizontal=True,
            key=f"ganador-{clave_form}",
        )
        clave = st.text_input("Contraseña", type="password", key=f"clave-{clave_form}")
        enviado = st.form_submit_button("Guardar partido", type="primary")

    if not enviado:
        return
    if ganador is None:
        st.error("Elegí quién ganó.")
        return
    try:
        ganador = GANADOR_POR_ETIQUETA[ganador]
        partido = armar_partido(formatear_fecha(dia), equipo1, equipo2, ganador)
        contenido = agregar_partido(texto_csv, partido)
    except ValueError as exc:
        st.error(str(exc))
        return
    aplicar_cambio(
        data_path,
        contenido,
        sha,
        clave,
        f"Post partido {partido['Fecha']} (cargado desde la app)",
        f"Partido del {partido['Fecha']} guardado: ganó {ETIQUETAS_GANADOR[ganador]}.",
    )


def mostrar_equipos_leidos(equipo1: list[str], equipo2: list[str]) -> None:
    col1, col2 = st.columns(2)
    for col, nombre, equipo in ((col1, "Clara", equipo1), (col2, "Oscura", equipo2)):
        with col:
            st.markdown(f"**{nombre}** ({len(equipo)})")
            st.markdown("\n".join(f"- {j}" for j in equipo))


def render_cargar_partido(
    df: pd.DataFrame, texto_csv: str, sha: str | None, data_path: Path
) -> None:
    st.subheader("Cargar partido")
    if config_github() is None:
        st.caption(
            "Modo local: los cambios se guardan en data.csv de esta máquina. "
            "Para que persistan en Streamlit Cloud configurá `[github]` en los secrets."
        )
    jugadores = obtener_jugadores(df)
    tab_pegar, tab_elegir, tab_editar = st.tabs(
        ["Pegar mensaje de WhatsApp", "Elegir jugadores", "Editar / borrar"]
    )

    with tab_pegar:
        mensaje = st.text_area(
            "Pegá el mensaje con los equipos",
            height=200,
            placeholder="Clara:\nJugador1;Jugador2;...\n\nOscura:\nJugadorA;JugadorB;...",
        )
        if mensaje.strip():
            try:
                equipo1, equipo2, avisos = parsear_mensaje_whatsapp(mensaje, jugadores)
            except ValueError as exc:
                st.error(str(exc))
            else:
                for aviso in avisos:
                    st.warning(aviso)
                mostrar_equipos_leidos(equipo1, equipo2)
                form_resultado("pegar", equipo1, equipo2, texto_csv, sha, data_path)

    with tab_elegir:
        equipo1 = st.multiselect("Clara (Equipo 1)", options=jugadores, key="equipo1_guiado")
        equipo2 = st.multiselect("Oscura (Equipo 2)", options=jugadores, key="equipo2_guiado")
        form_resultado("elegir", equipo1, equipo2, texto_csv, sha, data_path)

    with tab_editar:
        render_editar_partido(df, texto_csv, sha, data_path)


def render_editar_partido(
    df: pd.DataFrame, texto_csv: str, sha: str | None, data_path: Path
) -> None:
    if df.empty:
        st.info("No hay partidos cargados.")
        return

    def etiqueta(numero: int) -> str:
        fila = df.iloc[numero - 1]
        ganador = str(fila["Ganador"]).strip()
        return f"#{numero} · {fila['Fecha']} · ganó {ETIQUETAS_GANADOR.get(ganador, ganador)}"

    opciones = {etiqueta(n): n for n in range(len(df), 0, -1)}
    numero = opciones[st.selectbox("Partido", list(opciones), key="editar_numero")]
    fila = df.iloc[numero - 1]
    ganador_actual = str(fila["Ganador"]).strip()

    with st.form(f"form-editar-{numero}"):
        fecha = st.text_input("Fecha (d/m/aaaa)", value=str(fila["Fecha"]), key=f"ed-fecha-{numero}")
        equipo1_txt = st.text_area(
            "Clara (Equipo 1), separados por ;", value=fila["Equipo1"], key=f"ed-eq1-{numero}"
        )
        equipo2_txt = st.text_area(
            "Oscura (Equipo 2), separados por ;", value=fila["Equipo2"], key=f"ed-eq2-{numero}"
        )
        ganador = st.radio(
            "¿Quién ganó?",
            list(GANADOR_POR_ETIQUETA),
            index=GANADORES.index(ganador_actual) if ganador_actual in GANADORES else 0,
            horizontal=True,
            key=f"ed-ganador-{numero}",
        )
        clave = st.text_input("Contraseña", type="password", key=f"ed-clave-{numero}")
        confirmar_borrado = st.checkbox(
            "Confirmo que quiero borrar este partido", key=f"ed-confirmar-{numero}"
        )
        col1, col2 = st.columns(2)
        guardar = col1.form_submit_button("Guardar cambios", type="primary")
        borrar = col2.form_submit_button("Borrar partido")

    if guardar:
        try:
            partido = armar_partido(
                fecha,
                [j for j in parse_team(equipo1_txt) if j],
                [j for j in parse_team(equipo2_txt) if j],
                GANADOR_POR_ETIQUETA[ganador],
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        aplicar_cambio(
            data_path,
            editar_partido(texto_csv, numero, partido),
            sha,
            clave,
            f"Editado partido #{numero} ({partido['Fecha']}) desde la app",
            f"Partido #{numero} actualizado.",
        )
    elif borrar:
        if not confirmar_borrado:
            st.error("Marcá la confirmación para borrar.")
            return
        aplicar_cambio(
            data_path,
            borrar_partido(texto_csv, numero),
            sha,
            clave,
            f"Borrado partido #{numero} ({fila['Fecha']}) desde la app",
            f"Partido #{numero} ({fila['Fecha']}) borrado.",
        )


def main() -> None:
    st.set_page_config(page_title="Elo Fútbol", layout="wide")
    st.title("Dashboard Elo Fútbol")

    data_path = Path("data.csv")
    min_partidos = 5

    try:
        texto_csv, sha = leer_historial(data_path)
    except requests.RequestException as exc:
        st.error(f"No se pudo leer data.csv de GitHub: {exc}")
        return
    df = pd.read_csv(io.StringIO(texto_csv))
    st.session_state["data_path"] = data_path

    aviso = st.session_state.pop("aviso_guardado", None)
    if aviso:
        st.success(aviso)
    render_resumen_ultimo_partido(df)

    tabs = st.tabs(
        [
            "Historial",
            "Ranking",
            "Stats jugador",
            "Armar equipos",
            "Cargar partido",
        ]
    )

    with tabs[0]:
        render_historial(df)
    with tabs[1]:
        render_ranking(df, min_partidos, data_path)
    with tabs[2]:
        render_stats_jugador(df, data_path)
    with tabs[3]:
        render_equipos(df, data_path)
    with tabs[4]:
        render_cargar_partido(df, texto_csv, sha, data_path)


if __name__ == "__main__":
    main()
