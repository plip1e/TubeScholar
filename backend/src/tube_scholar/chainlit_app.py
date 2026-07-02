"""Chainlit UI — kept as a quick internal debug front-end during the migration.

The heavy lifting (graph, nodes, streaming filter) now lives in
``tube_scholar.core.graph``; this file is just the Chainlit glue that streams the
graph's answer tokens into a chat message.

Run from the project root:
    chainlit run backend/src/tube_scholar/chainlit_app.py

(Avoid the `-w` watch flag here: the SQLite checkpointer writes to data/ on every
turn, which the watcher would treat as a code change and reload the app mid-request.
Only use `-w` while actively editing this file's code.)
"""

import chainlit as cl

from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from tube_scholar.core.graph import get_graph, extract_text, is_answer_token


@cl.on_message
async def on_message(message: cl.Message):
    """Handle a user message by running the graph and streaming the final answer back."""
    user = cl.user_session.get("user")
    thread_id = user.identifier if user else cl.context.session.id
    config = {"configurable": {"thread_id": thread_id}}
    graph = await get_graph()
    final_answer = cl.Message(content="")

    # Note: we intentionally do NOT attach cl.LangchainCallbackHandler here. It traces
    # nested runs for Chainlit's step view, but is incompatible with LangGraph subgraph
    # streaming (floods logs with TracerException "No indexed run ID" / "Chat model
    # tracing is not supported"). We stream tokens directly from astream instead.
    #
    # subgraphs=True so the supervisor's tokens stream through; each item is
    # (namespace, (message_chunk, metadata)).
    async for namespace, (chunk, metadata) in graph.astream(
        {"messages": [HumanMessage(content=message.content)]},
        stream_mode="messages",
        subgraphs=True,
        config=RunnableConfig(**config),
    ):
        if is_answer_token(chunk, namespace, metadata):
            token = extract_text(chunk)
            if token:
                await final_answer.stream_token(token)

    await final_answer.send()
