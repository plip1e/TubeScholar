"""Chat endpoint.

Streams the LangGraph's answer tokens to the frontend using **Server-Sent Events
(SSE)**. This is the same graph the Chainlit UI uses — the only difference is the
transport: instead of Chainlit's ``stream_token``, we emit SSE events that the
browser reads with ``EventSource`` / ``fetch`` + ``ReadableStream``.

Wire protocol (each line is one SSE event):
    data: {"token": "..."}     # a piece of the answer
    ...
    data: {"done": true}       # stream finished
"""

import json

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from tube_scholar.core.graph import get_graph, extract_text, is_answer_token

# An APIRouter is a mini-app; main.py mounts it under the /chat prefix.
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Request body the frontend POSTs. FastAPI validates it via Pydantic."""

    message: str
    thread_id: str = "default"  # LangGraph checkpoint key (conversation memory)


async def _token_stream(message: str, thread_id: str):
    """Async generator yielding SSE events as the graph produces answer tokens."""
    graph = await get_graph()
    config = RunnableConfig(configurable={"thread_id": thread_id})

    # subgraphs=True so the supervisor's tokens stream through; each item is
    # (namespace, (message_chunk, metadata)).
    async for namespace, (chunk, metadata) in graph.astream(
        {"messages": [HumanMessage(content=message)]},
        stream_mode="messages",
        subgraphs=True,
        config=config,
    ):
        if is_answer_token(chunk, namespace, metadata):
            token = extract_text(chunk)
            if token:
                yield {"data": json.dumps({"token": token})}

    yield {"data": json.dumps({"done": True})}


@router.post("")
async def chat(req: ChatRequest) -> EventSourceResponse:
    """Stream the assistant's answer back to the client token-by-token over SSE."""
    return EventSourceResponse(_token_stream(req.message, req.thread_id))