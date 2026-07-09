"""FastAPI application entrypoint.

Run in dev with:
    uvicorn tube_scholar.main:app --reload --app-dir backend/src
or via the console script defined in pyproject.toml:
    tubescholar

Interactive API docs are auto-generated at http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tube_scholar import __version__
from tube_scholar.api import chat, ingest, videos

app = FastAPI(title="TubeScholar API", version=__version__)

# Allow the (separate) frontend dev server to call this API from the browser.
# In production you'd lock this down to your real domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default dev port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the routers. Each router already carries its own prefix (/chat, /ingest, /videos).
app.include_router(chat.router)
app.include_router(ingest.router)
app.include_router(videos.router)


@app.get("/health", tags=["meta"])
async def health() -> dict:
    """Liveness probe — handy for deployment platforms and a first smoke test."""
    return {"status": "ok", "version": __version__}


def main() -> None:
    """Console-script entrypoint (the `tubescholar` command)."""
    import uvicorn

    uvicorn.run("tube_scholar.main:app", host="0.0.0.0", port=8000, reload=True)