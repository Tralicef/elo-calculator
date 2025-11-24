# Proyecto Elo Fútbol

Este proyecto utiliza Python con las librerías trueskill y pandas para análisis de datos.

## Configuración del Entorno

1. Instalar uv (si no está instalado):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Crear y activar el entorno virtual (recomendado Python 3.11 para evitar builds de matplotlib):
```bash
uv venv --python 3.11
source .venv/bin/activate  # En Linux/Mac
# o
.venv\Scripts\activate  # En Windows
```

3. Instalar dependencias:
```bash
uv pip install -r requirements.txt
```

## Uso

### Configuración

1. Crea tu archivo de configuración copiando el ejemplo:
```bash
cp config.example.json config.json
```

2. Edita `config.json` con tus preferencias:
   - **jugadores_disponibles**: Lista de jugadores disponibles para el próximo partido
   - **restricciones_mismo_equipo**: Grupos de jugadores que deben jugar juntos (opcional)
   - **restricciones_distinto_equipo**: Grupos de jugadores que deben estar en equipos diferentes (opcional)
   - **jugadores_por_equipo**: Número de jugadores por equipo
   - **min_partidos_ranking**: Mínimo de partidos para aparecer en el ranking
   - **historial_path**: Ruta al archivo CSV con el historial de partidos

Ejemplo de `config.json`:
```json
{
  "jugadores_disponibles": ["Franco", "Walter", "Fede", ...],
  "restricciones_mismo_equipo": [["Franco", "Walter"]],
  "restricciones_distinto_equipo": [["Franco", "Hernán"]],
  "jugadores_por_equipo": 8,
  "min_partidos_ranking": 5,
  "historial_path": "data.csv"
}
```

### Ejecutar el calculador

```bash
uv run elo_calculator.py
```

El script:
1. Cargará la configuración desde `config.json`
2. Mostrará el ranking actual de jugadores
3. Generará equipos balanceados respetando las restricciones configuradas
4. Mostrará la calidad del partido propuesto

### Dashboard Streamlit

```bash
uv run streamlit run streamlit_app.py
```

## Características

- **Sistema de rating TrueSkill**: Calcula el nivel de cada jugador basándose en el historial
- **Equipos balanceados**: Crea equipos equilibrados para partidos competitivos
- **Restricciones**: Permite especificar jugadores que deben jugar juntos
- **Historial de partidos**: Actualiza automáticamente los ratings según los resultados 
