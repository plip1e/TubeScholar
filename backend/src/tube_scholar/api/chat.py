"""Chat endpoint.

Streams the LangGraph's answer tokens to the frontend using **Server-Sent Events
(SSE)**. This is the same graph the Chainlit UI uses; the only difference is the
transport: instead of Chainlit's ``stream_token``, we emit SSE events that the
browser reads with ``EventSource`` / ``fetch`` + ``ReadableStream``.

Wire protocol (each line is one SSE event):
    data: {"status": "..."}    # what the agent is doing right now (tool use, fact-check)
    data: {"token": "..."}     # a piece of the answer
    data: {"reset": true}      # verification forced a revision: discard shown text,
    ...                        #   a fresh answer streams next
    data: {"done": true}       # stream finished

The status events exist because a supervisor run can legitimately take minutes
(searching transcripts, ingesting videos, fact-checking). Without them the user
stares at a cursor with no way to tell "working hard" from "dead".
"""

import json

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from tube_scholar.core.graph import (
    get_graph,
    extract_text,
    is_answer_token,
    is_internal_namespace,
)

# An APIRouter is a mini-app; main.py mounts it under the /chat prefix.
router = APIRouter(prefix="/chat", tags=["chat"])

# Friendly one-liners for the supervisor's tools, shown while the tool runs.
TOOL_STATUS = {
    "search_transcripts": "Searching the video transcripts…",
    "search_transcripts_multi": "Searching the video transcripts…",
    "search_youtube": "Searching YouTube…",
    "ingest_video": "Ingesting a video (transcript + embeddings)…",
    "ingest_videos": "Ingesting videos (transcripts + embeddings)…",
    "get_video_info": "Looking up video details…",
    "list_videos": "Checking the video library…",
    "delete_video": "Updating the video library…",
    # the supervisor's handoff to the verification sub-agent (langgraph_supervisor
    # has used both naming schemes for handoff tools, so map both)
    "verification_agent": "Fact-checking the draft answer…",
    "transfer_to_verification_agent": "Fact-checking the draft answer…",
}


class ChatRequest(BaseModel):
    """Request body the frontend POSTs. FastAPI validates it via Pydantic."""

    message: str
    thread_id: str = "default"  # LangGraph checkpoint key (conversation memory)


def _status_for(chunk, namespace, metadata) -> str | None:
    """A user-facing status line for a non-answer chunk, or None.

    Two sources: the supervisor announcing a tool call (the chunk carries
    tool_call_chunks with the tool's name), and any activity inside the
    verification agent's subgraph (its content is filtered from the answer,
    but its presence means fact-checking is underway).
    """
    if is_internal_namespace(namespace):
        return TOOL_STATUS["transfer_to_verification_agent"]
    # Only the supervisor's react node ("agent") calls the real tools; this
    # gate keeps internal machinery (e.g. the intent classifier's structured
    # output, which is also a "tool call") from leaking into the status line.
    if metadata.get("langgraph_node") != "agent":
        return None
    for tool_call in getattr(chunk, "tool_call_chunks", None) or []:
        name = tool_call.get("name")
        if name:
            return TOOL_STATUS.get(name, f"Using {name}…")
    return None


async def _token_stream(message: str, thread_id: str):
    """Async generator yielding SSE events as the graph produces answer tokens."""
    graph = await get_graph()
    config = RunnableConfig(configurable={"thread_id": thread_id})

    last_status = None  # dedupe: only emit a status when it changes
    answer_id = None  # message id of the generation currently on screen

    # subgraphs=True so the supervisor's tokens stream through; each item is
    # (namespace, (message_chunk, metadata)).
    async for namespace, (chunk, metadata) in graph.astream(
        {"messages": [HumanMessage(content=message)]},
        stream_mode="messages",
        subgraphs=True,
        config=config,
    ):
        # NB: a chunk from the "agent" node can be an answer chunk with no
        # text — that's the supervisor *calling a tool* (the call itself
        # streams as an empty-content AIMessageChunk). Those must fall
        # through to the status check below, hence text-first ordering.
        token = extract_text(chunk) if is_answer_token(chunk, namespace, metadata) else ""
        if token:
            # All chunks of one LLM generation share a message id. A NEW id
            # while text is already on screen means the supervisor started a
            # fresh answer (a post-verification revision): tell the frontend
            # to clear the draft instead of appending a near-duplicate.
            chunk_id = getattr(chunk, "id", None)
            if answer_id is not None and chunk_id != answer_id:
                yield {"data": json.dumps({"reset": True})}
            answer_id = chunk_id
            last_status = None  # tokens supersede any status on screen
            yield {"data": json.dumps({"token": token})}
        else:
            status = _status_for(chunk, namespace, metadata)
            if status and status != last_status:
                last_status = status
                yield {"data": json.dumps({"status": status})}

    yield {"data": json.dumps({"done": True})}


@router.post("")
async def chat(req: ChatRequest) -> EventSourceResponse:
    """Stream the assistant's answer back to the client token-by-token over SSE."""
    return EventSourceResponse(_token_stream(req.message, req.thread_id))
