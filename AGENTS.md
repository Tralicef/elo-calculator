# Repository Guidelines

## Estructura del proyecto
- Raíz: scripts principales `elo_calculator.py` (cálculo de TrueSkill y armado de equipos) y `matriz_jugadores.py` (matrices y heatmaps).
- Datos: `data.csv` con historial de partidos; imágenes generadas (`*_rendimiento_*.jpg`, `matriz_jugadores.jpg`, etc.) quedan en la raíz.
- Configuración: `config.example.json` como plantilla; copia local en `config.json` (no versionar datos sensibles).
- Dependencias: `requirements.txt` para entorno Python 3.10+ (pandas, trueskill, seaborn, matplotlib, numpy).

## Configuración, build y ejecución
- Crear entorno virtual con `uv venv --python 3.11` y activar; instalar con `uv pip install -r requirements.txt` (3.11 evita compilar matplotlib).
- Preparar config: `cp config.example.json config.json` y ajustar rutas (`historial_path`) y restricciones.
- Ejecutar calculador: `uv run elo_calculator.py` (imprime ranking y sugiere equipos balanceados).
- Generar matrices/heatmaps: `uv run matriz_jugadores.py` tras tener `data.csv` actualizado; produce `matriz_jugadores.jpg` y variantes.
- Dashboard: `uv run streamlit run streamlit_app.py`.

## Estilo de código y nombres
- Python PEP8: 4 espacios, líneas claras, nombres en `snake_case`. Favor funciones puras y validaciones tempranas (patrón usado en `cargar_configuracion` y `crear_equipos_balanceados`).
- Documentar parámetros y retornos en docstrings breves; comentarios solo para lógica no evidente.
- Archivos de config JSON en minúsculas, claves en `snake_case`; mantener `utf-8` para nombres acentuados.

## Pruebas y validación
- No hay suite automática; valida corriendo `python elo_calculator.py` con un `config.json` pequeño y revisando que ranking y equipos se calculen sin excepciones.
- Para cambios en análisis de datos, prueba con subconjuntos de `data.csv` y verifica que las imágenes se generan (se guardan en raíz).
- Si agregas tests, usa `pytest` y ubícalos en `tests/` (no existe aún).

## Datos y seguridad
- `data.csv` puede contener información sensible del grupo; no compartirlo públicamente.
- No commitear `config.json` si contiene rutas locales o restricciones privadas; usar siempre `config.example.json` como referencia.
- Las imágenes derivan de datos; evita subirlas si contienen nombres que no deban hacerse públicos.

## Commits y PRs
- Mensajes concisos en español siguiendo el historial (`Actualizado…`, `Post partido…`, `Data until…`); usa modo imperativo o descripciones breves.
- Antes de abrir PR: describe propósito, adjunta comandos ejecutados, incluye muestras de salida (ranking/equipos) o capturas de gráficos si aplican.
- Referencia issues o TODOs en la descripción; mantiene cambios de datos y código en commits separados cuando sea posible.
