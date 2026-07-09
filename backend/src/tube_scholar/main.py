"""FastAPI application entrypoint.

Run in dev with:
    uvicorn tube_scholar.main:app --reload --app-dir backend/src
or via the console script defined in pyproject.toml:
    tubescholar

Interactive API docs are auto-generated at http://localhost:8000/docs

Deployment shape: ONE server does everything. All API routes live under the
``/api`` prefix, and if a built frontend exists (``frontend/dist``, made by
``npm run build``) it is served at ``/`` from this same app. Frontend and API
being same-origin means no CORS configuration is needed anywhere:

    dev:   browser -> Vite :5173 (serves the frontend, proxies /api -> :8000)
    prod:  browser -> uvicorn :$PORT (serves the frontend AND /api)
"""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from tube_scholar import __version__
from tube_scholar.api import chat, ingest, videos
from tube_scholar.core.config import settings

app = FastAPI(title="TubeScholar API", version=__version__)

# Mount the routers under /api. Each router carries its own sub-prefix, so the
# full paths are /api/chat, /api/ingest, /api/videos.
app.include_router(chat.router, prefix="/api")
app.include_router(ingest.router, prefix="/api")
app.include_router(videos.router, prefix="/api")


@app.get("/api/health", tags=["meta"])
async def health() -> dict:
    """Liveness probe, handy for deployment platforms and a first smoke test."""
    return {"status": "ok", "version": __version__}


# Serve the built frontend, when there is one. In dev there usually isn't
# (Vite serves it on :5173), so this mount simply doesn't happen. Mounting "/"
# LAST matters: FastAPI tries the routes above first, so /api/* and /docs keep
# working; everything else falls through to the static files. html=True makes
# "/" serve index.html.
_static_dir = Path(settings.static_dir)
if _static_dir.is_dir():
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="frontend")


def main() -> None:
    """Console-script entrypoint (the `tubescholar` command).

    Port comes from the PORT env var (deployment platforms set it; Hugging
    Face Spaces expects 7860). No --reload here: reload is a dev-server
    feature; use the uvicorn command in the module docstring while developing.
    """
    import uvicorn

    uvicorn.run(
        "tube_scholar.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
