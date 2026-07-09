"""Video-ingestion endpoint.

Wraps the shared VideoIngestionPipeline: accept a YouTube URL, pull the
transcript, embed it, store it in Chroma. Uses the same lazy pipeline singleton
as the graph (``get_pipeline``), so a video ingested here is immediately
searchable by the agent: one store, two doors.

The pipeline is synchronous (network + embedding calls), so the handler runs it
in FastAPI's threadpool via ``run_in_threadpool``, since otherwise it would block
the event loop and freeze every other request (including in-flight chat streams).
"""

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from tube_scholar.core.graph import get_pipeline

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestRequest(BaseModel):
    url: str


@router.post("")
async def ingest_video(req: IngestRequest) -> dict:
    """Ingest one video. Always returns the pipeline's status dict (``status`` is
    ``ingested``/``skipped`` on success, or a named failure like ``invalid_url``/
    ``transcript_unavailable``) so the frontend can react without try/except."""
    pipeline = await run_in_threadpool(get_pipeline)
    return await run_in_threadpool(pipeline.ingest_video, req.url)
