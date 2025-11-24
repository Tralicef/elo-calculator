import pandas as pd
from trueskill import Rating, rate, quality, setup
from collections import defaultdict
from itertools import combinations
import random
import json
import os

setup(draw_probability=0.1)  # permite empates


def parse_team(team_str):
    return [name.strip() for name in team_str.split(";")]


def cargar_configuracion(config_path="config.json"):
    """
    Carga la configuración desde un archivo JSON.

    Args:
        config_path: Ruta al archivo de configuración JSON

    Returns:
        dict: Diccionario con la configuración cargada
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración '{config_path}'. "
            "Por favor, crea un archivo config.json con la configuración necesaria."
        )
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Validar campos requeridos
    campos_requeridos = ['jugadores_disponibles', 'jugadores_por_equipo']
    for campo in campos_requeridos:
        if campo not in config:
            raise ValueError(f"El archivo de configuración debe contener el campo '{campo}'")
    
    # Establecer valores por defecto para campos opcionales
    config.setdefault('restricciones_mismo_equipo', [])
    config.setdefault('restricciones_distinto_equipo', [])
    config.setdefault('min_partidos_ranking', 5)
    config.setdefault('historial_path', 'data.csv')
    
    return config


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


def crear_equipos_balanceados(jugadores, ratings, jugadores_por_equipo=5, restricciones_mismo_equipo=None, restricciones_distinto_equipo=None):
    """
    Crea dos equipos balanceados a partir de una lista de jugadores.

    Args:
        jugadores: Lista de nombres de jugadores disponibles
        ratings: Diccionario con los ratings actuales
        jugadores_por_equipo: Número de jugadores por equipo
        restricciones_mismo_equipo: Lista de listas con jugadores que deben jugar juntos en el mismo equipo
                                    Ejemplo: [["Franco", "Walter"], ["Fede", "José MB"]]
        restricciones_distinto_equipo: Lista de listas con jugadores que deben estar en equipos diferentes
                                       Ejemplo: [["Franco", "Hernán"], ["Fede", "Nacho"]]

    Returns:
        tuple: (equipo1, equipo2, calidad_del_partido)
    """
    if len(jugadores) < jugadores_por_equipo * 2:
        raise ValueError(f"Se necesitan al menos {jugadores_por_equipo * 2} jugadores")

    # Obtener ratings de los jugadores disponibles
    jugadores_ratings = {j: ratings.get(j, Rating()) for j in jugadores}

    # Validar restricciones de mismo equipo
    if restricciones_mismo_equipo is None:
        restricciones_mismo_equipo = []
    
    # Validar restricciones de distinto equipo
    if restricciones_distinto_equipo is None:
        restricciones_distinto_equipo = []
    
    # Verificar que todos los jugadores en restricciones estén disponibles
    for grupo in restricciones_mismo_equipo:
        for jugador in grupo:
            if jugador not in jugadores:
                raise ValueError(f"El jugador '{jugador}' en las restricciones de mismo equipo no está disponible")
    
    for grupo in restricciones_distinto_equipo:
        for jugador in grupo:
            if jugador not in jugadores:
                raise ValueError(f"El jugador '{jugador}' en las restricciones de distinto equipo no está disponible")
    
    # Verificar que no haya jugadores duplicados entre grupos de mismo equipo
    todos_restringidos_mismo = [j for grupo in restricciones_mismo_equipo for j in grupo]
    if len(todos_restringidos_mismo) != len(set(todos_restringidos_mismo)):
        raise ValueError("Un jugador no puede estar en múltiples grupos de restricciones de mismo equipo")
    
    # Verificar que las restricciones no sean contradictorias
    for grupo in restricciones_distinto_equipo:
        # Verificar si hay jugadores de este grupo que deben estar juntos
        for grupo_mismo in restricciones_mismo_equipo:
            jugadores_en_ambos = set(grupo) & set(grupo_mismo)
            if len(jugadores_en_ambos) >= 2:
                raise ValueError(
                    f"Conflicto de restricciones: {list(jugadores_en_ambos)} deben estar juntos "
                    f"pero también en equipos diferentes"
                )

    def cumple_restricciones(equipo):
        """Verifica si un equipo cumple con todas las restricciones"""
        # Verificar restricciones de mismo equipo
        for grupo in restricciones_mismo_equipo:
            # Contar cuántos jugadores del grupo están en este equipo
            en_equipo = sum(1 for j in grupo if j in equipo)
            # Si algún jugador del grupo está en el equipo, todos deben estar
            if en_equipo > 0 and en_equipo != len(grupo):
                return False
        return True
    
    def cumple_restricciones_distinto(equipo1, equipo2):
        """Verifica si dos equipos cumplen con las restricciones de estar en equipos diferentes"""
        for grupo in restricciones_distinto_equipo:
            # Contar cuántos del grupo están en cada equipo
            en_equipo1 = sum(1 for j in grupo if j in equipo1)
            en_equipo2 = sum(1 for j in grupo if j in equipo2)
            # Todos deben estar distribuidos (no todos en el mismo equipo)
            if en_equipo1 == len(grupo) or en_equipo2 == len(grupo):
                return False
        return True

    # Generar todas las combinaciones posibles de equipos
    mejores_equipos = None
    mejor_calidad = -1

    for equipo1 in combinations(jugadores, jugadores_por_equipo):
        equipo1 = list(equipo1)
        equipo2 = [j for j in jugadores if j not in equipo1]

        # Verificar que ambos equipos cumplan con las restricciones de mismo equipo
        if not cumple_restricciones(equipo1) or not cumple_restricciones(equipo2):
            continue
        
        # Verificar que cumplan con las restricciones de distinto equipo
        if not cumple_restricciones_distinto(equipo1, equipo2):
            continue

        # Calcular calidad del partido
        r1 = [jugadores_ratings[j] for j in equipo1]
        r2 = [jugadores_ratings[j] for j in equipo2]
        calidad = quality([r1, r2])

        if calidad > mejor_calidad:
            mejor_calidad = calidad
            mejores_equipos = (equipo1, equipo2)

    if mejores_equipos is None:
        raise ValueError("No se pudo crear equipos que cumplan con las restricciones especificadas")

    return mejores_equipos[0], mejores_equipos[1], mejor_calidad


def proximo_partido(
    jugadores_disponibles,
    historial_path="data.csv",
    min_partidos=5,
    jugadores_por_equipo=9,
    restricciones_mismo_equipo=None,
    restricciones_distinto_equipo=None,
):
    """
    Prepara el próximo partido con los jugadores disponibles.

    Args:
        jugadores_disponibles: Lista de nombres de jugadores disponibles
        historial_path: Ruta al archivo CSV con el historial
        min_partidos: Número mínimo de partidos para mostrar en el ranking
        jugadores_por_equipo: Número de jugadores por equipo
        restricciones_mismo_equipo: Lista de listas con jugadores que deben jugar juntos en el mismo equipo
                                    Ejemplo: [["Franco", "Walter"], ["Fede", "José MB"]]
        restricciones_distinto_equipo: Lista de listas con jugadores que deben estar en equipos diferentes
                                       Ejemplo: [["Franco", "Hernán"], ["Fede", "Nacho"]]

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
    print("\nRanking actual (mínimo 5 partidos):")
    for jugador, mu, sigma, partidos, victorias in mostrar_ranking(
        ratings, historial_path, min_partidos
    ):
        print(f"{jugador}: {mu} ± {sigma} ({victorias}/{partidos} victorias)")

    # Mostrar restricciones si existen
    if restricciones_mismo_equipo or restricciones_distinto_equipo:
        print("\nRestricciones aplicadas:")
        if restricciones_mismo_equipo:
            print("  ✅ Mismo equipo (deben jugar juntos):")
            for i, grupo in enumerate(restricciones_mismo_equipo, 1):
                print(f"     {i}. {', '.join(grupo)}")
        if restricciones_distinto_equipo:
            print("  ❌ Distinto equipo (deben estar separados):")
            for i, grupo in enumerate(restricciones_distinto_equipo, 1):
                print(f"     {i}. {', '.join(grupo)}")

    # Crear equipos balanceados
    equipo1, equipo2, calidad = crear_equipos_balanceados(
        jugadores_disponibles, ratings, jugadores_por_equipo, 
        restricciones_mismo_equipo, restricciones_distinto_equipo
    )

    print("\nEquipos balanceados sugeridos:")
    print(f"Equipo 1: {'; '.join(equipo1)}")
    print(f"Equipo 2: {'; '.join(equipo2)}")
    print(f"Calidad del partido: {calidad:.2%}")

    return equipo1, equipo2, calidad


# Ejemplo de uso
if __name__ == "__main__":
    # Cargar configuración desde el archivo config.json
    try:
        config = cargar_configuracion()
        
        print("=== Configuración cargada ===")
        print(f"Jugadores disponibles: {len(config['jugadores_disponibles'])}")
        print(f"Jugadores por equipo: {config['jugadores_por_equipo']}")
        print(f"Mínimo de partidos para ranking: {config['min_partidos_ranking']}")
        
        total_restricciones = (len(config['restricciones_mismo_equipo']) + 
                              len(config['restricciones_distinto_equipo']))
        if total_restricciones > 0:
            print(f"Restricciones configuradas: {total_restricciones} grupo(s)")
            if config['restricciones_mismo_equipo']:
                print(f"  - Mismo equipo: {len(config['restricciones_mismo_equipo'])}")
            if config['restricciones_distinto_equipo']:
                print(f"  - Distinto equipo: {len(config['restricciones_distinto_equipo'])}")
        print("=" * 30)
        
        # Preparar próximo partido usando la configuración
        proximo_partido(
            jugadores_disponibles=config['jugadores_disponibles'],
            historial_path=config['historial_path'],
            min_partidos=config['min_partidos_ranking'],
            jugadores_por_equipo=config['jugadores_por_equipo'],
            restricciones_mismo_equipo=config['restricciones_mismo_equipo'],
            restricciones_distinto_equipo=config['restricciones_distinto_equipo']
        )
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPor favor, crea un archivo 'config.json' con el siguiente formato:")
        print("""
{
  "jugadores_disponibles": ["Jugador1", "Jugador2", ...],
  "restricciones_mismo_equipo": [["Jugador1", "Jugador2"]],
  "restricciones_distinto_equipo": [["Jugador3", "Jugador4"]],
  "jugadores_por_equipo": 8,
  "min_partidos_ranking": 5,
  "historial_path": "data.csv"
}
        """)
    except Exception as e:
        print(f"Error: {e}")
