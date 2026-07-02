"""Video-ingestion endpoint (skeleton).

Wraps the existing VideoIngestionPipeline: accept a YouTube URL, pull the
transcript, embed it, store it in Chroma. Stubbed for now.
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestRequest(BaseModel):
    url: str


@router.post("")
async def ingest_video(req: IngestRequest) -> dict:
    # TODO: call VideoIngestionPipeline once the logic is moved into this package.
    return {"status": "stub", "url": req.url}