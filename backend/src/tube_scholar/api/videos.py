"""Read-only listing of ingested videos.

Backs the frontend's video sidebar. Reads from the same pipeline registry the
agent tools use (rebuilt from Chroma on startup), so it reflects everything
ingested, whether via POST /ingest or by the agent mid-conversation.
"""

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from tube_scholar.core.graph import get_pipeline

router = APIRouter(prefix="/videos", tags=["videos"])


@router.get("")
async def list_videos() -> list[dict]:
    """Every ingested video's metadata, ordered by placement.

    First call constructs the pipeline (Chroma + embeddings init, which blocks),
    hence the threadpool; after that it's an in-memory dict read.
    """
    pipeline = await run_in_threadpool(get_pipeline)
    return pipeline.list_videos()
