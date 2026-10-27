"""Lectura y escritura de data.csv en GitHub para que los cambios persistan en Streamlit Cloud."""
from __future__ import annotations

import base64

import requests

API_URL = "https://api.github.com"


class ConflictoError(Exception):
    """El archivo cambió en GitHub desde que se leyó (otro lo modificó en paralelo)."""


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def leer_archivo(token: str, repo: str, ruta: str, rama: str) -> tuple[str, str]:
    """
    Lee un archivo de texto del repo.

    Returns:
        tuple: (contenido, sha) donde sha identifica la versión leída
    """
    resp = requests.get(
        f"{API_URL}/repos/{repo}/contents/{ruta}",
        headers=_headers(token),
        params={"ref": rama},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return base64.b64decode(data["content"]).decode("utf-8"), data["sha"]


def guardar_archivo(
    token: str, repo: str, ruta: str, rama: str, contenido: str, sha: str, mensaje: str
) -> str:
    """
    Commitea el contenido nuevo del archivo.

    Args:
        sha: Versión sobre la que se hicieron los cambios; si el archivo cambió
             desde entonces, GitHub rechaza el commit

    Returns:
        str: sha de la versión nueva

    Raises:
        ConflictoError: Si alguien modificó el archivo en el medio
    """
    resp = requests.put(
        f"{API_URL}/repos/{repo}/contents/{ruta}",
        headers=_headers(token),
        json={
            "message": mensaje,
            "content": base64.b64encode(contenido.encode("utf-8")).decode("ascii"),
            "sha": sha,
            "branch": rama,
        },
        timeout=15,
    )
    if resp.status_code == 409:
        raise ConflictoError("data.csv cambió mientras editabas. Recargá la página y probá de nuevo.")
    resp.raise_for_status()
    return resp.json()["content"]["sha"]
