import pandas as pd
from trueskill import Rating, rate, quality, setup
from collections import defaultdict
from itertools import combinations
import random

setup(draw_probability=0.1)  # permite empates


def parse_team(team_str):
    return [name.strip() for name in team_str.split(";")]


def contar_partidos_y_victorias(historial_path):
    """
    Cuenta cuántos partidos ha jugado y ganado cada jugador.

    Args:
        historial_path: Ruta al archivo CSV con el historial de partidos

    Returns:
        tuple: (dict_partidos, dict_victorias) donde cada diccionario tiene jugador como clave
    """
    df = pd.read_csv(historial_path)
    partidos = defaultdict(int)
    victorias = defaultdict(int)

    for _, row in df.iterrows():
        t1 = parse_team(row["Equipo1"])
        t2 = parse_team(row["Equipo2"])

        # Contar partidos
        for jugador in t1 + t2:
            partidos[jugador] += 1

        # Contar victorias
        if row["Ganador"] == "Equipo1":
            for jugador in t1:
                victorias[jugador] += 1
        elif row["Ganador"] == "Equipo2":
            for jugador in t2:
                victorias[jugador] += 1
        # En caso de empate, no se cuenta como victoria para nadie

    return dict(partidos), dict(victorias)


def actualizar_elo(historial_path):
    df = pd.read_csv(historial_path)
    ratings = defaultdict(Rating)

    for _, row in df.iterrows():
        t1 = parse_team(row["Equipo1"])
        t2 = parse_team(row["Equipo2"])
        r1 = [ratings[p] for p in t1]
        r2 = [ratings[p] for p in t2]

        if row["Ganador"] == "Equipo1":
            ranks = [0, 1]
        elif row["Ganador"] == "Equipo2":
            ranks = [1, 0]
        else:
            ranks = [0, 0]  # empate

        new_r1, new_r2 = rate([r1, r2], ranks=ranks)
        for p, r in zip(t1, new_r1):
            ratings[p] = r
        for p, r in zip(t2, new_r2):
            ratings[p] = r

    return dict(ratings)


def mostrar_ranking(ratings, historial_path=None, min_partidos=0):
    """
    Muestra el ranking de jugadores, opcionalmente filtrado por mínimo de partidos.

    Args:
        ratings: Diccionario con los ratings de los jugadores
        historial_path: Ruta al archivo CSV con el historial (opcional)
        min_partidos: Número mínimo de partidos para mostrar al jugador

    Returns:
        list: Lista ordenada de tuplas (jugador, mu, sigma, partidos, victorias)
    """
    if min_partidos > 0 and historial_path:
        partidos, victorias = contar_partidos_y_victorias(historial_path)
        jugadores_filtrados = {
            p: r for p, r in ratings.items() if partidos.get(p, 0) >= min_partidos
        }
    else:
        jugadores_filtrados = ratings
        partidos = {}
        victorias = {}

    return sorted(
        [
            (
                p,
                round(r.mu, 1),
                round(r.sigma, 1),
                partidos.get(p, 0),
                victorias.get(p, 0),
            )
            for p, r in jugadores_filtrados.items()
        ],
        key=lambda x: x[1],
        reverse=True,
    )


def crear_equipos_balanceados(jugadores, ratings, jugadores_por_equipo=5):
    """
    Crea dos equipos balanceados a partir de una lista de jugadores.

    Args:
        jugadores: Lista de nombres de jugadores disponibles
        ratings: Diccionario con los ratings actuales
        jugadores_por_equipo: Número de jugadores por equipo

    Returns:
        tuple: (equipo1, equipo2, calidad_del_partido)
    """
    if len(jugadores) < jugadores_por_equipo * 2:
        raise ValueError(f"Se necesitan al menos {jugadores_por_equipo * 2} jugadores")

    # Obtener ratings de los jugadores disponibles
    jugadores_ratings = {j: ratings.get(j, Rating()) for j in jugadores}

    # Generar todas las combinaciones posibles de equipos
    mejores_equipos = None
    mejor_calidad = -1

    for equipo1 in combinations(jugadores, jugadores_por_equipo):
        equipo1 = list(equipo1)
        equipo2 = [j for j in jugadores if j not in equipo1]

        # Calcular calidad del partido
        r1 = [jugadores_ratings[j] for j in equipo1]
        r2 = [jugadores_ratings[j] for j in equipo2]
        calidad = quality([r1, r2])

        if calidad > mejor_calidad:
            mejor_calidad = calidad
            mejores_equipos = (equipo1, equipo2)

    return mejores_equipos[0], mejores_equipos[1], mejor_calidad


def proximo_partido(
    jugadores_disponibles,
    historial_path="data.csv",
    min_partidos=2,
    jugadores_por_equipo=8,
):
    """
    Prepara el próximo partido con los jugadores disponibles.

    Args:
        jugadores_disponibles: Lista de nombres de jugadores disponibles
        historial_path: Ruta al archivo CSV con el historial
        min_partidos: Número mínimo de partidos para mostrar en el ranking
        jugadores_por_equipo: Número de jugadores por equipo

    Returns:
        tuple: (equipo1, equipo2, calidad) si hay suficientes jugadores, None en caso contrario
    """
    if not jugadores_disponibles:
        print("No hay jugadores disponibles para el próximo partido.")
        return None

    if len(jugadores_disponibles) < jugadores_por_equipo * 2:
        print(
            f"Se necesitan al menos {jugadores_por_equipo * 2} jugadores para formar los equipos."
        )
        return None

    # Obtener ratings actuales
    ratings = actualizar_elo(historial_path)

    # Mostrar ranking completo (solo filtrado por mínimo de partidos)
    print("\nRanking actual (mínimo 2 partidos):")
    for jugador, mu, sigma, partidos, victorias in mostrar_ranking(
        ratings, historial_path, min_partidos
    ):
        print(f"{jugador}: {mu} ± {sigma} ({victorias}/{partidos} victorias)")

    # Crear equipos balanceados
    equipo1, equipo2, calidad = crear_equipos_balanceados(
        jugadores_disponibles, ratings, jugadores_por_equipo
    )

    print("\nEquipos balanceados sugeridos:")
    print(f"Equipo 1: {'; '.join(equipo1)}")
    print(f"Equipo 2: {'; '.join(equipo2)}")
    print(f"Calidad del partido: {calidad:.2%}")

    return equipo1, equipo2, calidad


# Ejemplo de uso
if __name__ == "__main__":
    # Lista de jugadores disponibles para el próximo partido
    jugadores = [
        "Franco",
        "Nacho R",
        "Leo",
        "José MB",
        "Fer MB",
        "Gonza",
        "José A",
        "Hernán",
        "Fer F",
        "Santi DLV",
        "Ponja",
        "Nacho",
        "Diego",
        "Gael",
        "René",
        "Juan C"
        ]

    # Preparar próximo partido
    proximo_partido(jugadores)
