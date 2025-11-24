import pandas as pd
import numpy as np
from collections import defaultdict
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from itertools import combinations

def procesar_datos_partidos(archivo_csv):
    """
    Procesa el archivo CSV de partidos y extrae todos los jugadores
    """
    df = pd.read_csv(archivo_csv)
    
    # Diccionario para contar partidos por jugador
    partidos_por_jugador = defaultdict(int)
    # Lista para almacenar todas las combinaciones de jugadores por partido
    combinaciones_partidos = []
    
    for _, fila in df.iterrows():
        # Extraer jugadores del Equipo1
        equipo1 = [jugador.strip() for jugador in fila['Equipo1'].split(';')]
        # Extraer jugadores del Equipo2
        equipo2 = [jugador.strip() for jugador in fila['Equipo2'].split(';')]
        
        # Contar partidos por jugador
        for jugador in equipo1 + equipo2:
            partidos_por_jugador[jugador] += 1
        
        # Agregar todas las combinaciones de jugadores del mismo equipo
        combinaciones_partidos.append(equipo1)
        combinaciones_partidos.append(equipo2)
    
    return partidos_por_jugador, combinaciones_partidos

def crear_matriz_jugadores(partidos_por_jugador, combinaciones_partidos, min_partidos=2):
    """
    Crea una matriz que muestra cuántas veces jugó cada jugador con otro
    """
    # Filtrar jugadores que jugaron al menos 2 veces
    jugadores_filtrados = [jugador for jugador, partidos in partidos_por_jugador.items() 
                          if partidos >= min_partidos]
    
    # Ordenar alfabéticamente
    jugadores_filtrados.sort()
    
    # Crear matriz vacía
    n = len(jugadores_filtrados)
    matriz = np.zeros((n, n), dtype=int)
    
    # Diccionario para mapear nombres a índices
    jugador_a_indice = {jugador: i for i, jugador in enumerate(jugadores_filtrados)}
    
    # Contar partidos juntos
    for partido in combinaciones_partidos:
        for i, jugador1 in enumerate(partido):
            if jugador1 in jugador_a_indice:
                for j, jugador2 in enumerate(partido):
                    if i != j and jugador2 in jugador_a_indice:
                        idx1 = jugador_a_indice[jugador1]
                        idx2 = jugador_a_indice[jugador2]
                        matriz[idx1][idx2] += 1
    
    return matriz, jugadores_filtrados

def crear_matriz_ponderada_correcta(partidos_por_jugador, combinaciones_partidos, jugadores):
    """
    Crea una matriz ponderada que muestra fracciones:
    veces que jugaron en el mismo equipo / veces que estuvieron en el mismo partido
    """
    # Crear matriz para almacenar las fracciones como texto
    matriz_fracciones = np.empty((len(jugadores), len(jugadores)), dtype=object)
    # Crear matriz numérica para los colores del heatmap
    matriz_numerica = np.zeros((len(jugadores), len(jugadores)), dtype=float)
    
    # Diccionario para mapear nombres a índices
    jugador_a_indice = {jugador: i for i, jugador in enumerate(jugadores)}
    
    # Contar partidos juntos y partidos en el mismo partido
    partidos_mismo_equipo = defaultdict(int)  # (jugador1, jugador2) -> count
    partidos_mismo_partido = defaultdict(int)  # (jugador1, jugador2) -> count
    
    # Recorrer cada partido
    for equipo in combinaciones_partidos:
        # Para cada par de jugadores en el mismo equipo
        for i in range(len(equipo)):
            for j in range(i+1, len(equipo)):
                jugador1, jugador2 = equipo[i], equipo[j]
                if jugador1 in jugador_a_indice and jugador2 in jugador_a_indice:
                    # Incrementar contador de mismo equipo (en ambas direcciones)
                    partidos_mismo_equipo[(jugador1, jugador2)] += 1
                    partidos_mismo_equipo[(jugador2, jugador1)] += 1
                    # También incrementar contador de mismo partido
                    partidos_mismo_partido[(jugador1, jugador2)] += 1
                    partidos_mismo_partido[(jugador2, jugador1)] += 1
    
    # Ahora necesitamos contar cuando estuvieron en el mismo partido pero en equipos diferentes
    # Recorrer los partidos de a pares (Equipo1 vs Equipo2)
    for idx in range(0, len(combinaciones_partidos), 2):
        if idx + 1 < len(combinaciones_partidos):
            equipo1 = combinaciones_partidos[idx]
            equipo2 = combinaciones_partidos[idx + 1]
            
            # Para cada par de jugadores (uno en cada equipo)
            for jugador1 in equipo1:
                for jugador2 in equipo2:
                    if jugador1 in jugador_a_indice and jugador2 in jugador_a_indice:
                        # Incrementar contador de mismo partido (rivales)
                        partidos_mismo_partido[(jugador1, jugador2)] += 1
                        partidos_mismo_partido[(jugador2, jugador1)] += 1
    
    # Llenar las matrices
    for i, jugador1 in enumerate(jugadores):
        for j, jugador2 in enumerate(jugadores):
            if i != j:
                mismo_equipo = partidos_mismo_equipo.get((jugador1, jugador2), 0)
                mismo_partido = partidos_mismo_partido.get((jugador1, jugador2), 0)
                
                if mismo_partido > 0:
                    matriz_fracciones[i][j] = f"{mismo_equipo}/{mismo_partido}"
                    matriz_numerica[i][j] = (mismo_equipo / mismo_partido) * 100
                else:
                    matriz_fracciones[i][j] = ""
                    matriz_numerica[i][j] = np.nan
            else:
                matriz_fracciones[i][j] = ""
                matriz_numerica[i][j] = np.nan
    
    return matriz_fracciones, matriz_numerica

def aplicar_formato_condicional(matriz, jugadores):
    """
    Aplica formato condicional a la matriz usando pandas
    """
    df = pd.DataFrame(matriz, index=jugadores, columns=jugadores)
    
    # Función para aplicar formato condicional
    def formatear_celda(valor):
        if valor == 0:
            return 'background-color: #f0f0f0; color: #999999'
        elif valor == 1:
            return 'background-color: #e8f5e8; color: #2d5a2d'
        elif valor == 2:
            return 'background-color: #d4edda; color: #155724'
        elif valor == 3:
            return 'background-color: #c3e6cb; color: #0f5132'
        elif valor == 4:
            return 'background-color: #b1dfbb; color: #0a3628'
        else:
            return 'background-color: #a8e6cf; color: #0a3628; font-weight: bold'
    
    return df.style.map(formatear_celda)

def crear_imagen_matriz_seaborn(matriz, jugadores, nombre_archivo='matriz_jugadores.jpg'):
    """
    Crea una imagen JPG de la matriz usando seaborn
    """
    # Configurar el estilo de seaborn
    sns.set_theme(style="white")
    
    # Crear DataFrame para seaborn
    df_matriz = pd.DataFrame(matriz, index=jugadores, columns=jugadores)
    
    # Configurar la figura
    plt.figure(figsize=(16, 12))
    
    # Crear el heatmap con seaborn
    ax = sns.heatmap(df_matriz, 
                     annot=True,  # Mostrar valores en las celdas
                     fmt='d',      # Formato decimal
                     cmap='Greens', # Paleta de colores verde
                     cbar_kws={'label': 'Partidos jugados juntos'},
                     square=True,   # Celdas cuadradas
                     linewidths=0.5, # Grosor de las líneas
                     linecolor='white') # Color de las líneas
    
    # Configurar título y etiquetas
    plt.title('Matriz de Jugadores - Partidos Jugados Juntos', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Jugadores', fontsize=12, fontweight='bold')
    plt.ylabel('Jugadores', fontsize=12, fontweight='bold')
    
    # Rotar etiquetas del eje X para mejor legibilidad
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Ajustar layout para evitar cortes
    plt.tight_layout()
    
    # Guardar imagen
    plt.savefig(nombre_archivo, 
                dpi=300,           # Alta resolución
                bbox_inches='tight', # Sin márgenes extra
                format='jpg')
    
    plt.close()  # Cerrar la figura para liberar memoria
    
    return nombre_archivo

def crear_imagen_matriz_ponderada(matriz_fracciones, matriz_numerica, jugadores, nombre_archivo='matriz_ponderada.jpg'):
    """
    Crea una imagen JPG de la matriz de fracciones usando seaborn
    """
    # Configurar el estilo de seaborn
    sns.set_theme(style="white")
    
    # Crear DataFrame para seaborn (usar matriz numérica para colores)
    df_matriz = pd.DataFrame(matriz_numerica, index=jugadores, columns=jugadores)
    
    # Configurar la figura
    plt.figure(figsize=(18, 14))
    
    # Crear el heatmap con seaborn
    ax = sns.heatmap(df_matriz, 
                     annot=matriz_fracciones,  # Mostrar fracciones como texto
                     fmt='',       # Formato vacío para texto personalizado
                     cmap='RdYlBu_r', # Paleta de colores divergente
                     cbar_kws={'label': 'Afinidad (porcentaje)'},
                     square=True,   # Celdas cuadradas
                     linewidths=0.5, # Grosor de las líneas
                     linecolor='white', # Color de las líneas
                     center=50,     # Centro de la paleta en 50%
                     vmin=0,        # Valor mínimo
                     vmax=100,      # Valor máximo (porcentaje)
                     annot_kws={'size': 8}) # Tamaño del texto
    
    # Configurar título y etiquetas
    plt.title('Matriz de Afinidad - Fracciones de Partidos\n(Veces juntos / Veces mismo partido)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Jugadores', fontsize=12, fontweight='bold')
    plt.ylabel('Jugadores', fontsize=12, fontweight='bold')
    
    # Rotar etiquetas del eje X para mejor legibilidad
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Ajustar layout para evitar cortes
    plt.tight_layout()
    
    # Guardar imagen
    plt.savefig(nombre_archivo, 
                dpi=300,           # Alta resolución
                bbox_inches='tight', # Sin márgenes extra
                format='jpg')
    
    plt.close()  # Cerrar la figura para liberar memoria
    
    return nombre_archivo

def crear_grafico_red_conexiones(matriz, jugadores, nombre_archivo='red_conexiones.jpg'):
    """
    Crea un gráfico de red que muestra las conexiones entre jugadores
    """
    try:
        import networkx as nx
        
        # Crear grafo
        G = nx.Graph()
        
        # Agregar nodos (jugadores)
        for jugador in jugadores:
            G.add_node(jugador)
        
        # Agregar aristas (conexiones) solo si jugaron juntos
        for i in range(len(jugadores)):
            for j in range(i+1, len(jugadores)):
                if matriz[i][j] > 0:
                    G.add_edge(jugadores[i], jugadores[j], weight=matriz[i][j])
        
        # Configurar la figura
        plt.figure(figsize=(20, 16))
        
        # Posiciones de los nodos usando layout de resorte
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Dibujar el grafo
        nx.draw_networkx_nodes(G, pos, 
                              node_color='lightblue', 
                              node_size=1000,
                              alpha=0.8)
        
        # Dibujar las aristas con grosor proporcional al peso
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, 
                              width=[w/2 for w in weights], 
                              edge_color='gray',
                              alpha=0.6)
        
        # Dibujar etiquetas
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        # Título
        plt.title('Red de Conexiones entre Jugadores\n(El grosor de las líneas indica partidos juntos)', 
                  fontsize=16, fontweight='bold', pad=20)
        
        # Ajustar layout
        plt.axis('off')
        plt.tight_layout()
        
        # Guardar imagen
        plt.savefig(nombre_archivo, 
                    dpi=300, 
                    bbox_inches='tight', 
                    format='jpg')
        
        plt.close()
        return nombre_archivo
        
    except ImportError:
        print("❌ NetworkX no está instalado. Instala con: pip install networkx")
        return None

def crear_grafico_top_duplas_fracciones(matriz_fracciones, matriz_numerica, jugadores, nombre_archivo='top_duplas.jpg'):
    """
    Crea un gráfico de barras con las mejores duplas basado en fracciones
    Solo considera duplas que estuvieron en al menos 5 partidos juntos
    """
    # Encontrar las mejores duplas
    duplas = []
    for i in range(len(jugadores)):
        for j in range(i+1, len(jugadores)):
            if matriz_fracciones[i][j] and matriz_fracciones[i][j] != "":
                # Extraer numerador y denominador
                fraccion = matriz_fracciones[i][j]
                numerador, denominador = map(int, fraccion.split('/'))
                
                # Filtrar: solo si estuvieron en al menos 5 partidos
                if denominador >= 5:
                    porcentaje = matriz_numerica[i][j]
                    duplas.append((jugadores[i], jugadores[j], fraccion, porcentaje, numerador, denominador))
    
    # Ordenar por porcentaje de afinidad (descendente)
    duplas.sort(key=lambda x: x[3], reverse=True)
    
    # Tomar las top 15
    top_duplas = duplas[:15]
    
    # Preparar datos para el gráfico
    nombres_duplas = [f"{d[0]} + {d[1]}" for d in top_duplas]
    porcentajes = [d[3] for d in top_duplas]
    fracciones = [d[2] for d in top_duplas]
    
    # Configurar la figura
    plt.figure(figsize=(16, 10))
    
    # Crear gráfico de barras con colores según porcentaje
    colors = plt.cm.RdYlGn([p/100 for p in porcentajes])
    bars = plt.barh(range(len(top_duplas)), porcentajes, 
                    color=colors, alpha=0.8, edgecolor='navy')
    
    # Configurar ejes
    plt.yticks(range(len(top_duplas)), nombres_duplas)
    plt.xlabel('Afinidad (%)', fontsize=12, fontweight='bold')
    plt.title('Top 15 Duplas - Mayor Afinidad\n(Mínimo 5 partidos juntos)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Agregar valores en las barras (fracción y porcentaje)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f"{fracciones[i]} ({width:.1f}%)", 
                ha='left', va='center', fontweight='bold', fontsize=9)
    
    # Invertir eje Y para mostrar la mejor dupla arriba
    plt.gca().invert_yaxis()
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar imagen
    plt.savefig(nombre_archivo, 
                dpi=300, 
                bbox_inches='tight', 
                format='jpg')
    
    plt.close()
    
    return nombre_archivo

def crear_grafico_peores_duplas_fracciones(matriz_fracciones, matriz_numerica, jugadores, nombre_archivo='peores_duplas.jpg'):
    """
    Crea un gráfico de barras con las peores duplas (menor afinidad)
    Solo considera duplas que estuvieron en al menos 5 partidos juntos
    """
    # Encontrar las peores duplas
    duplas = []
    for i in range(len(jugadores)):
        for j in range(i+1, len(jugadores)):
            if matriz_fracciones[i][j] and matriz_fracciones[i][j] != "":
                # Extraer numerador y denominador
                fraccion = matriz_fracciones[i][j]
                numerador, denominador = map(int, fraccion.split('/'))
                
                # Filtrar: solo si estuvieron en al menos 5 partidos
                if denominador >= 5:
                    porcentaje = matriz_numerica[i][j]
                    duplas.append((jugadores[i], jugadores[j], fraccion, porcentaje, numerador, denominador))
    
    # Ordenar por porcentaje de afinidad (ascendente)
    duplas.sort(key=lambda x: x[3])
    
    # Tomar las peores 15
    peores_duplas = duplas[:15]
    
    # Preparar datos para el gráfico
    nombres_duplas = [f"{d[0]} + {d[1]}" for d in peores_duplas]
    porcentajes = [d[3] for d in peores_duplas]
    fracciones = [d[2] for d in peores_duplas]
    
    # Configurar la figura
    plt.figure(figsize=(16, 10))
    
    # Crear gráfico de barras con colores según porcentaje (invertido)
    colors = plt.cm.RdYlGn([p/100 for p in porcentajes])
    bars = plt.barh(range(len(peores_duplas)), porcentajes, 
                    color=colors, alpha=0.8, edgecolor='navy')
    
    # Configurar ejes
    plt.yticks(range(len(peores_duplas)), nombres_duplas)
    plt.xlabel('Afinidad (%)', fontsize=12, fontweight='bold')
    plt.title('Top 15 Duplas - Menor Afinidad (Rivales Frecuentes)\n(Mínimo 5 partidos juntos)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Agregar valores en las barras (fracción y porcentaje)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f"{fracciones[i]} ({width:.1f}%)", 
                ha='left', va='center', fontweight='bold', fontsize=9)
    
    # Invertir eje Y para mostrar la peor dupla arriba
    plt.gca().invert_yaxis()
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar imagen
    plt.savefig(nombre_archivo, 
                dpi=300, 
                bbox_inches='tight', 
                format='jpg')
    
    plt.close()
    
    return nombre_archivo

def calcular_rendimiento_duplas(archivo_csv, min_partidos=5):
    """
    Calcula el rendimiento de las duplas basado en porcentaje de victorias cuando jugaron juntas.
    """
    df = pd.read_csv(archivo_csv)
    estadisticas = defaultdict(lambda: {"jugados": 0, "ganados": 0})

    for _, fila in df.iterrows():
        equipo1 = [jugador.strip() for jugador in fila['Equipo1'].split(';')]
        equipo2 = [jugador.strip() for jugador in fila['Equipo2'].split(';')]
        ganador = fila['Ganador']

        for dupla in combinations(sorted(equipo1), 2):
            estadisticas[dupla]["jugados"] += 1
            if ganador == "Equipo1":
                estadisticas[dupla]["ganados"] += 1

        for dupla in combinations(sorted(equipo2), 2):
            estadisticas[dupla]["jugados"] += 1
            if ganador == "Equipo2":
                estadisticas[dupla]["ganados"] += 1

    ranking = []
    for (jugador1, jugador2), datos in estadisticas.items():
        jugados = datos["jugados"]
        ganados = datos["ganados"]
        if jugados >= min_partidos:
            porcentaje = (ganados / jugados) * 100 if jugados > 0 else 0.0
            ranking.append((jugador1, jugador2, ganados, jugados, porcentaje))

    ranking.sort(key=lambda x: (x[4], x[3], x[2]), reverse=True)
    return ranking

def crear_grafico_duplas_rendimiento(ranking, nombre_archivo='duplas_rendimiento_top.jpg', top_n=15, mejores=True, min_partidos=4):
    """
    Genera un gráfico de barras con el porcentaje de victorias de las duplas.
    """
    if not ranking:
        print("⚠️ No hay suficientes datos para generar el ranking de duplas por rendimiento.")
        return None

    if mejores:
        duplas_seleccionadas = ranking[:top_n]
        cmap = plt.cm.Greens
        titulo = f"Top {len(duplas_seleccionadas)} Duplas - Porcentaje de Victorias\n(Mínimo {min_partidos} partidos juntos)"
    else:
        ranking_peores = sorted(ranking, key=lambda x: (x[4], -x[3], -x[2]))
        duplas_seleccionadas = ranking_peores[:top_n]
        cmap = plt.cm.Reds
        titulo = f"Peores {len(duplas_seleccionadas)} Duplas - Porcentaje de Victorias\n(Mínimo {min_partidos} partidos juntos)"

    duplas_seleccionadas = [dupla for dupla in duplas_seleccionadas if dupla[3] > 0]
    if not duplas_seleccionadas:
        print("⚠️ No hay duplas con partidos jugados para graficar el rendimiento.")
        return None

    nombres = [f"{d[0]} + {d[1]}" for d in duplas_seleccionadas]
    porcentajes = [d[4] for d in duplas_seleccionadas]
    registros = [f"{d[2]}/{d[3]}" for d in duplas_seleccionadas]

    plt.figure(figsize=(16, 10))
    porcentajes_normalizados = [np.clip(p / 100, 0, 1) for p in porcentajes]
    colores = cmap(porcentajes_normalizados)
    barras = plt.barh(range(len(duplas_seleccionadas)), porcentajes, color=colores, alpha=0.85, edgecolor='navy')

    plt.yticks(range(len(duplas_seleccionadas)), nombres)
    plt.xlabel('Porcentaje de victorias (%)', fontsize=12, fontweight='bold')
    plt.title(titulo, fontsize=16, fontweight='bold', pad=20)

    for i, barra in enumerate(barras):
        valor = barra.get_width()
        plt.text(valor + 1, barra.get_y() + barra.get_height() / 2,
                 f"{registros[i]} ({valor:.1f}%)",
                 ha='left', va='center', fontweight='bold', fontsize=9)

    plt.gca().invert_yaxis()

    plt.xlim(0, max(105, max(porcentajes) + 5))
    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight', format='jpg')
    plt.close()

    return nombre_archivo

def calcular_rendimiento_trios(archivo_csv, min_partidos=5):
    """
    Calcula el rendimiento de los tríos basado en porcentaje de victorias cuando jugaron juntos.
    """
    df = pd.read_csv(archivo_csv)
    estadisticas = defaultdict(lambda: {"jugados": 0, "ganados": 0})

    for _, fila in df.iterrows():
        equipo1 = [jugador.strip() for jugador in fila['Equipo1'].split(';')]
        equipo2 = [jugador.strip() for jugador in fila['Equipo2'].split(';')]
        ganador = fila['Ganador']

        if len(equipo1) >= 3:
            for trio in combinations(sorted(equipo1), 3):
                estadisticas[trio]["jugados"] += 1
                if ganador == "Equipo1":
                    estadisticas[trio]["ganados"] += 1

        if len(equipo2) >= 3:
            for trio in combinations(sorted(equipo2), 3):
                estadisticas[trio]["jugados"] += 1
                if ganador == "Equipo2":
                    estadisticas[trio]["ganados"] += 1

    ranking = []
    for trio, datos in estadisticas.items():
        jugados = datos["jugados"]
        ganados = datos["ganados"]
        if jugados >= min_partidos:
            porcentaje = (ganados / jugados) * 100 if jugados > 0 else 0.0
            ranking.append((*trio, ganados, jugados, porcentaje))

    ranking.sort(key=lambda x: (x[5], x[3], x[4]), reverse=True)
    return ranking

def crear_grafico_trios_rendimiento(ranking, nombre_archivo='trios_rendimiento_top.jpg', top_n=10, mejores=True, min_partidos=5):
    """
    Genera un gráfico de barras con el porcentaje de victorias de los tríos.
    """
    if not ranking:
        print("⚠️ No hay suficientes datos para generar el ranking de tríos por rendimiento.")
        return None

    if mejores:
        trios_seleccionados = ranking[:top_n]
        cmap = plt.cm.Greens
        titulo = f"Top {len(trios_seleccionados)} Tríos - Porcentaje de Victorias\n(Mínimo {min_partidos} partidos juntos)"
    else:
        ranking_peores = sorted(ranking, key=lambda x: (x[5], -x[4], -x[3]))
        trios_seleccionados = ranking_peores[:top_n]
        cmap = plt.cm.Reds
        titulo = f"Peores {len(trios_seleccionados)} Tríos - Porcentaje de Victorias\n(Mínimo {min_partidos} partidos juntos)"

    trios_seleccionados = [trio for trio in trios_seleccionados if trio[4] > 0]
    if not trios_seleccionados:
        print("⚠️ No hay tríos con partidos jugados para graficar el rendimiento.")
        return None

    nombres = [f"{t[0]} + {t[1]} + {t[2]}" for t in trios_seleccionados]
    porcentajes = [t[5] for t in trios_seleccionados]
    registros = [f"{t[3]}/{t[4]}" for t in trios_seleccionados]  # ganados/jugados

    plt.figure(figsize=(16, 10))
    porcentajes_normalizados = [np.clip(p / 100, 0, 1) for p in porcentajes]
    colores = cmap(porcentajes_normalizados)
    barras = plt.barh(range(len(trios_seleccionados)), porcentajes, color=colores, alpha=0.85, edgecolor='navy')

    plt.yticks(range(len(trios_seleccionados)), nombres, fontsize=9)
    plt.xlabel('Porcentaje de victorias (%)', fontsize=12, fontweight='bold')
    plt.title(titulo, fontsize=16, fontweight='bold', pad=20)

    for i, barra in enumerate(barras):
        valor = barra.get_width()
        ganados, jugados = trios_seleccionados[i][3], trios_seleccionados[i][4]
        plt.text(valor + 1, barra.get_y() + barra.get_height() / 2,
                 f"{ganados}/{jugados} ({valor:.1f}%)",
                 ha='left', va='center', fontweight='bold', fontsize=8)

    plt.gca().invert_yaxis()
    plt.xlim(0, max(105, max(porcentajes) + 5))
    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight', format='jpg')
    plt.close()

    return nombre_archivo

def calcular_rendimiento_cuartetos(archivo_csv, min_partidos=3):
    """
    Calcula el rendimiento de los cuartetos basado en porcentaje de victorias cuando jugaron juntos.
    """
    df = pd.read_csv(archivo_csv)
    estadisticas = defaultdict(lambda: {"jugados": 0, "ganados": 0})

    for _, fila in df.iterrows():
        equipo1 = [jugador.strip() for jugador in fila['Equipo1'].split(';')]
        equipo2 = [jugador.strip() for jugador in fila['Equipo2'].split(';')]
        ganador = fila['Ganador']

        if len(equipo1) >= 4:
            for cuarteto in combinations(sorted(equipo1), 4):
                estadisticas[cuarteto]["jugados"] += 1
                if ganador == "Equipo1":
                    estadisticas[cuarteto]["ganados"] += 1

        if len(equipo2) >= 4:
            for cuarteto in combinations(sorted(equipo2), 4):
                estadisticas[cuarteto]["jugados"] += 1
                if ganador == "Equipo2":
                    estadisticas[cuarteto]["ganados"] += 1

    ranking = []
    for cuarteto, datos in estadisticas.items():
        jugados = datos["jugados"]
        ganados = datos["ganados"]
        if jugados >= min_partidos:
            porcentaje = (ganados / jugados) * 100 if jugados > 0 else 0.0
            ranking.append((*cuarteto, ganados, jugados, porcentaje))

    ranking.sort(key=lambda x: (x[6], x[4], x[5]), reverse=True)
    return ranking

def crear_grafico_cuartetos_rendimiento(ranking, nombre_archivo='cuartetos_rendimiento_top.jpg', top_n=8, mejores=True, min_partidos=3):
    """
    Genera un gráfico de barras con el porcentaje de victorias de los cuartetos.
    """
    if not ranking:
        print("⚠️ No hay suficientes datos para generar el ranking de cuartetos por rendimiento.")
        return None

    if mejores:
        cuartetos_seleccionados = ranking[:top_n]
        cmap = plt.cm.Greens
        titulo = f"Top {len(cuartetos_seleccionados)} Cuartetos - Porcentaje de Victorias\n(Mínimo {min_partidos} partidos juntos)"
    else:
        ranking_peores = sorted(ranking, key=lambda x: (x[6], -x[5], -x[4]))
        cuartetos_seleccionados = ranking_peores[:top_n]
        cmap = plt.cm.Reds
        titulo = f"Peores {len(cuartetos_seleccionados)} Cuartetos - Porcentaje de Victorias\n(Mínimo {min_partidos} partidos juntos)"

    cuartetos_seleccionados = [cuarteto for cuarteto in cuartetos_seleccionados if cuarteto[5] > 0]
    if not cuartetos_seleccionados:
        print("⚠️ No hay cuartetos con partidos jugados para graficar el rendimiento.")
        return None

    nombres = [f"{c[0]} + {c[1]} + {c[2]} + {c[3]}" for c in cuartetos_seleccionados]
    porcentajes = [c[6] for c in cuartetos_seleccionados]
    registros = [f"{c[4]}/{c[5]}" for c in cuartetos_seleccionados]

    plt.figure(figsize=(16, 10))
    porcentajes_normalizados = [np.clip(p / 100, 0, 1) for p in porcentajes]
    colores = cmap(porcentajes_normalizados)
    barras = plt.barh(range(len(cuartetos_seleccionados)), porcentajes, color=colores, alpha=0.85, edgecolor='navy')

    plt.yticks(range(len(cuartetos_seleccionados)), nombres, fontsize=9)
    plt.xlabel('Porcentaje de victorias (%)', fontsize=12, fontweight='bold')
    plt.title(titulo, fontsize=16, fontweight='bold', pad=20)

    for i, barra in enumerate(barras):
        valor = barra.get_width()
        ganados, jugados = cuartetos_seleccionados[i][4], cuartetos_seleccionados[i][5]
        plt.text(valor + 1, barra.get_y() + barra.get_height() / 2,
                 f"{ganados}/{jugados} ({valor:.1f}%)",
                 ha='left', va='center', fontweight='bold', fontsize=8)

    plt.gca().invert_yaxis()
    plt.xlim(0, max(105, max(porcentajes) + 5))
    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight', format='jpg')
    plt.close()

    return nombre_archivo

def calcular_rachas_jugadores(archivo_csv, ultimos_n=10):
    """
    Calcula las mejores y peores rachas (victorias y derrotas consecutivas) en los últimos N partidos.

    Returns:
        tuple:
            - lista de tuplas (jugador, mejor_racha, racha_actual, partidos_analizados)
            - lista de tuplas (jugador, peor_racha, racha_actual, partidos_analizados)
            - diccionario con estadísticas completas por jugador
    """
    df = pd.read_csv(archivo_csv)

    # Asegurar orden cronológico
    try:
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        df = df.sort_values('Fecha')
    except Exception:
        df = df.reset_index(drop=True)

    if ultimos_n is not None and ultimos_n > 0:
        df = df.tail(ultimos_n)

    rachas = defaultdict(lambda: {"actual": 0, "mejor": 0, "peor": 0, "partidos": 0})

    for _, fila in df.iterrows():
        equipo1 = [jugador.strip() for jugador in fila['Equipo1'].split(';')]
        equipo2 = [jugador.strip() for jugador in fila['Equipo2'].split(';')]
        ganador = fila['Ganador']

        jugadores_partido = equipo1 + equipo2
        for jugador in jugadores_partido:
            rachas[jugador]["partidos"] += 1

        if ganador == "Empate":
            for jugador in jugadores_partido:
                rachas[jugador]["actual"] = 0
            continue

        ganadores = equipo1 if ganador == "Equipo1" else equipo2
        perdedores = equipo2 if ganador == "Equipo1" else equipo1

        for jugador in ganadores:
            datos = rachas[jugador]
            datos["actual"] = datos["actual"] + 1 if datos["actual"] > 0 else 1
            datos["mejor"] = max(datos["mejor"], datos["actual"])

        for jugador in perdedores:
            datos = rachas[jugador]
            datos["actual"] = datos["actual"] - 1 if datos["actual"] < 0 else -1
            datos["peor"] = min(datos["peor"], datos["actual"])

    mejores = [
        (jugador, datos["mejor"], datos["actual"], datos["partidos"])
        for jugador, datos in rachas.items()
        if datos["mejor"] > 0
    ]
    peores = [
        (jugador, abs(datos["peor"]), datos["actual"], datos["partidos"])
        for jugador, datos in rachas.items()
        if datos["peor"] < 0
    ]

    mejores.sort(key=lambda x: (x[1], x[3]), reverse=True)
    peores.sort(key=lambda x: (x[1], x[3]), reverse=True)

    return mejores, peores, rachas

def main():
    print("Analizando partidos de fútbol...")
    print("=" * 50)
    
    # Procesar datos
    partidos_por_jugador, combinaciones_partidos = procesar_datos_partidos('data.csv')
    
    # Mostrar estadísticas básicas
    print(f"Total de jugadores únicos: {len(partidos_por_jugador)}")
    print(f"Total de partidos: {len(combinaciones_partidos)}")
    print()
    
    # Mostrar jugadores que jugaron al menos 5 veces
    jugadores_plus = [(jugador, partidos) for jugador, partidos in partidos_por_jugador.items() 
                        if partidos >= 5]
    jugadores_plus.sort(key=lambda x: x[1], reverse=True)
    
    print("Jugadores que jugaron al menos 5 veces:")
    print("-" * 40)
    for jugador, partidos in jugadores_plus:
        print(f"{jugador}: {partidos} partidos")
    print()
    
    # Crear matriz solo con jugadores que jugaron al menos 5 veces
    matriz, jugadores = crear_matriz_jugadores(partidos_por_jugador, combinaciones_partidos, min_partidos=5)
    
    print(f"Matriz de {len(jugadores)} jugadores que jugaron al menos 5 veces:")
    print("=" * 80)
    
    # Crear DataFrame con formato
    df_formateado = aplicar_formato_condicional(matriz, jugadores)
    
    # Mostrar matriz
    print(df_formateado.to_string())
    
    # Guardar en Excel con formato
    print("\nGuardando matriz en Excel...")
    with pd.ExcelWriter('matriz_jugadores.xlsx', engine='openpyxl') as writer:
        df_formateado.to_excel(writer, sheet_name='Matriz_Jugadores', index=True)
    
    print("✅ Matriz guardada en 'matriz_jugadores.xlsx'")
    
    # Crear imagen JPG con seaborn
    print("\nGenerando imagen JPG de la matriz con seaborn...")
    try:
        nombre_imagen = crear_imagen_matriz_seaborn(matriz, jugadores)
        print(f"✅ Imagen guardada como '{nombre_imagen}'")
    except Exception as e:
        print(f"❌ Error al generar imagen: {e}")
        print("Asegúrate de tener seaborn y matplotlib instalados correctamente")
    
    # Crear matriz ponderada
    print("\nGenerando matriz ponderada...")
    matriz_fracciones, matriz_numerica = crear_matriz_ponderada_correcta(partidos_por_jugador, combinaciones_partidos, jugadores)
    
    # Crear imagen de matriz ponderada
    print("Generando imagen JPG de la matriz ponderada...")
    try:
        nombre_imagen_ponderada = crear_imagen_matriz_ponderada(matriz_fracciones, matriz_numerica, jugadores)
        print(f"✅ Matriz ponderada guardada como '{nombre_imagen_ponderada}'")
    except Exception as e:
        print(f"❌ Error al generar matriz ponderada: {e}")
    
    # Crear gráfico de red
    print("\nGenerando gráfico de red de conexiones...")
    try:
        nombre_red = crear_grafico_red_conexiones(matriz, jugadores)
        if nombre_red:
            print(f"✅ Gráfico de red guardado como '{nombre_red}'")
    except Exception as e:
        print(f"❌ Error al generar gráfico de red: {e}")
    
    # Crear gráfico de top duplas con fracciones
    print("\nGenerando gráfico de top duplas (con fracciones)...")
    try:
        nombre_top_duplas = crear_grafico_top_duplas_fracciones(matriz_fracciones, matriz_numerica, jugadores)
        print(f"✅ Gráfico de top duplas guardado como '{nombre_top_duplas}'")
    except Exception as e:
        print(f"❌ Error al generar gráfico de top duplas: {e}")
    
    # Crear gráfico de peores duplas con fracciones
    print("\nGenerando gráfico de peores duplas (rivales frecuentes)...")
    try:
        nombre_peores_duplas = crear_grafico_peores_duplas_fracciones(matriz_fracciones, matriz_numerica, jugadores)
        print(f"✅ Gráfico de peores duplas guardado como '{nombre_peores_duplas}'")
    except Exception as e:
        print(f"❌ Error al generar gráfico de peores duplas: {e}")
    
    # Ranking de duplas por rendimiento
    print("\nCalculando ranking de duplas por porcentaje de victorias...")
    min_partidos_duplas = 4
    ranking_rendimiento = calcular_rendimiento_duplas('data.csv', min_partidos=min_partidos_duplas)

    if ranking_rendimiento:
        print(f"Top 10 duplas con mejor porcentaje de victorias (mínimo {min_partidos_duplas} partidos juntos):")
        for jugador1, jugador2, ganados, jugados, porcentaje in ranking_rendimiento[:10]:
            print(f"  {jugador1} + {jugador2}: {porcentaje:.1f}% ({ganados}/{jugados})")

        ranking_rendimiento_peores = sorted(ranking_rendimiento, key=lambda x: (x[4], -x[3], -x[2]))
        print(f"\nPeores 10 duplas en porcentaje de victorias (mínimo {min_partidos_duplas} partidos juntos):")
        for jugador1, jugador2, ganados, jugados, porcentaje in ranking_rendimiento_peores[:10]:
            print(f"  {jugador1} + {jugador2}: {porcentaje:.1f}% ({ganados}/{jugados})")

        print("\nGenerando gráficos de rendimiento de duplas...")
        try:
            nombre_top_rendimiento = crear_grafico_duplas_rendimiento(
                ranking_rendimiento,
                nombre_archivo='duplas_rendimiento_top.jpg',
                top_n=15,
                mejores=True,
                min_partidos=min_partidos_duplas
            )
            if nombre_top_rendimiento:
                print(f"✅ Gráfico de mejores duplas por rendimiento guardado como '{nombre_top_rendimiento}'")
        except Exception as e:
            print(f"❌ Error al generar gráfico de mejores duplas por rendimiento: {e}")

        try:
            nombre_peor_rendimiento = crear_grafico_duplas_rendimiento(
                ranking_rendimiento,
                nombre_archivo='duplas_rendimiento_peores.jpg',
                top_n=15,
                mejores=False,
                min_partidos=min_partidos_duplas
            )
            if nombre_peor_rendimiento:
                print(f"✅ Gráfico de peores duplas por rendimiento guardado como '{nombre_peor_rendimiento}'")
        except Exception as e:
            print(f"❌ Error al generar gráfico de peores duplas por rendimiento: {e}")
    else:
        print(f"No se encontraron duplas con al menos {min_partidos_duplas} partidos juntas para calcular el rendimiento.")
    
    # Ranking de tríos por rendimiento
    min_partidos_trios = 4
    ranking_trios = calcular_rendimiento_trios('data.csv', min_partidos=min_partidos_trios)

    if ranking_trios:
        print(f"\nTop 10 tríos con mejor porcentaje de victorias (mínimo {min_partidos_trios} partidos juntos):")
        for trio in ranking_trios[:10]:
            jugador1, jugador2, jugador3, ganados, jugados, porcentaje = trio
            print(f"  {jugador1} + {jugador2} + {jugador3}: {porcentaje:.1f}% ({ganados}/{jugados})")

        ranking_trios_peores = sorted(ranking_trios, key=lambda x: (x[5], -x[4], -x[3]))
        print(f"\nPeores 10 tríos en porcentaje de victorias (mínimo {min_partidos_trios} partidos juntos):")
        for trio in ranking_trios_peores[:10]:
            jugador1, jugador2, jugador3, ganados, jugados, porcentaje = trio
            print(f"  {jugador1} + {jugador2} + {jugador3}: {porcentaje:.1f}% ({ganados}/{jugados})")

        print("\nGenerando gráficos de rendimiento de tríos...")
        try:
            nombre_top_trios = crear_grafico_trios_rendimiento(
                ranking_trios,
                nombre_archivo='trios_rendimiento_top.jpg',
                top_n=10,
                mejores=True,
                min_partidos=min_partidos_trios
            )
            if nombre_top_trios:
                print(f"✅ Gráfico de mejores tríos por rendimiento guardado como '{nombre_top_trios}'")
        except Exception as e:
            print(f"❌ Error al generar gráfico de mejores tríos por rendimiento: {e}")

        try:
            nombre_peores_trios = crear_grafico_trios_rendimiento(
                ranking_trios,
                nombre_archivo='trios_rendimiento_peores.jpg',
                top_n=10,
                mejores=False,
                min_partidos=min_partidos_trios
            )
            if nombre_peores_trios:
                print(f"✅ Gráfico de peores tríos por rendimiento guardado como '{nombre_peores_trios}'")
        except Exception as e:
            print(f"❌ Error al generar gráfico de peores tríos por rendimiento: {e}")
    else:
        print(f"\nNo se encontraron tríos con al menos {min_partidos_trios} partidos juntos para calcular el rendimiento.")
    
    # Ranking de cuartetos por rendimiento
    min_partidos_cuartetos = 3
    ranking_cuartetos = calcular_rendimiento_cuartetos('data.csv', min_partidos=min_partidos_cuartetos)

    if ranking_cuartetos:
        print(f"\nTop 8 cuartetos con mejor porcentaje de victorias (mínimo {min_partidos_cuartetos} partidos juntos):")
        for cuarteto in ranking_cuartetos[:8]:
            j1, j2, j3, j4, ganados, jugados, porcentaje = cuarteto
            print(f"  {j1} + {j2} + {j3} + {j4}: {porcentaje:.1f}% ({ganados}/{jugados})")

        ranking_cuartetos_peores = sorted(ranking_cuartetos, key=lambda x: (x[6], -x[5], -x[4]))
        print(f"\nPeores 8 cuartetos en porcentaje de victorias (mínimo {min_partidos_cuartetos} partidos juntos):")
        for cuarteto in ranking_cuartetos_peores[:8]:
            j1, j2, j3, j4, ganados, jugados, porcentaje = cuarteto
            print(f"  {j1} + {j2} + {j3} + {j4}: {porcentaje:.1f}% ({ganados}/{jugados})")

        print("\nGenerando gráficos de rendimiento de cuartetos...")
        try:
            nombre_top_cuartetos = crear_grafico_cuartetos_rendimiento(
                ranking_cuartetos,
                nombre_archivo='cuartetos_rendimiento_top.jpg',
                top_n=8,
                mejores=True,
                min_partidos=min_partidos_cuartetos
            )
            if nombre_top_cuartetos:
                print(f"✅ Gráfico de mejores cuartetos por rendimiento guardado como '{nombre_top_cuartetos}'")
        except Exception as e:
            print(f"❌ Error al generar gráfico de mejores cuartetos por rendimiento: {e}")

        try:
            nombre_peores_cuartetos = crear_grafico_cuartetos_rendimiento(
                ranking_cuartetos,
                nombre_archivo='cuartetos_rendimiento_peores.jpg',
                top_n=8,
                mejores=False,
                min_partidos=min_partidos_cuartetos
            )
            if nombre_peores_cuartetos:
                print(f"✅ Gráfico de peores cuartetos por rendimiento guardado como '{nombre_peores_cuartetos}'")
        except Exception as e:
            print(f"❌ Error al generar gráfico de peores cuartetos por rendimiento: {e}")
    else:
        print(f"\nNo se encontraron cuartetos con al menos {min_partidos_cuartetos} partidos juntos para calcular el rendimiento.")
    
    # Rachas de jugadores
    ultimos_partidos_rachas = 10
    print(f"\nAnalizando rachas individuales en los últimos {ultimos_partidos_rachas} partidos...")
    mejores_rachas, peores_rachas, rachas_detalle = calcular_rachas_jugadores(
        'data.csv',
        ultimos_n=ultimos_partidos_rachas
    )

    if mejores_rachas:
        print(f"Mejores rachas de victorias (top 5):")
        for jugador, mejor_racha, racha_actual, partidos in mejores_rachas[:5]:
            print(
                f"  {jugador}: {mejor_racha} victorias consecutivas "
                f"(racha actual: {racha_actual:+d}, partidos analizados: {partidos})"
            )
    else:
        print("No se registraron rachas positivas en el período analizado.")

    if peores_rachas:
        print(f"\nPeores rachas de derrotas (top 5):")
        for jugador, peor_racha, racha_actual, partidos in peores_rachas[:5]:
            print(
                f"  {jugador}: {peor_racha} derrotas consecutivas "
                f"(racha actual: {racha_actual:+d}, partidos analizados: {partidos})"
            )
    else:
        print("No se registraron rachas negativas en el período analizado.")
    
    # Mostrar algunas estadísticas interesantes
    print("\n" + "=" * 50)
    print("ESTADÍSTICAS INTERESANTES:")
    print("=" * 50)
    
    # Encontrar la dupla que más jugó junta
    max_partidos = 0
    mejor_dupla = None
    
    for i in range(len(jugadores)):
        for j in range(i+1, len(jugadores)):
            if matriz[i][j] > max_partidos:
                max_partidos = matriz[i][j]
                mejor_dupla = (jugadores[i], jugadores[j])
    
    if mejor_dupla:
        print(f"Dupla que más jugó junta: {mejor_dupla[0]} y {mejor_dupla[1]} ({max_partidos} veces)")
    
    # Mostrar jugadores que jugaron con más gente diferente
    jugadores_conexiones = []
    for i, jugador in enumerate(jugadores):
        conexiones = sum(1 for j in range(len(jugadores)) if matriz[i][j] > 0)
        jugadores_conexiones.append((jugador, conexiones))
    
    jugadores_conexiones.sort(key=lambda x: x[1], reverse=True)
    print(f"\nJugadores con más conexiones (jugaron con más gente diferente):")
    for jugador, conexiones in jugadores_conexiones[:5]:
        print(f"  {jugador}: {conexiones} conexiones")
    
    # Mostrar estadísticas de la matriz ponderada
    valores_no_cero = matriz_numerica[~np.isnan(matriz_numerica)]
    if len(valores_no_cero) > 0:
        print(f"\nEstadísticas de Afinidad entre Jugadores:")
        print(f"  Promedio de afinidad: {np.mean(valores_no_cero):.1f}%")
        print(f"  Máximo afinidad: {np.max(matriz_numerica):.1f}%")
        print(f"  Mínimo afinidad (excluyendo 0): {np.min(valores_no_cero):.1f}%")
        print(f"  Total de conexiones analizadas: {len(valores_no_cero)}")
        print(f"  Interpretación: 100% = siempre juntos, 0% = nunca juntos")
        print(f"  Ejemplo: 3/10 = 30% de afinidad")
    else:
        print(f"\nNo se encontraron conexiones para analizar en la matriz de afinidad.")

if __name__ == "__main__":
    main()
