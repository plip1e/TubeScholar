"""LangGraph assembly for TubeScholar.

This is the brain of the app, lifted out of the old Chainlit ``app.py``. It builds:

    START -> classify -> route -> (chitchat | supervisor) -> END

Everything is built **lazily** (on first ``get_graph()`` call) rather than at import
time, so importing this module — and therefore booting the FastAPI app — is cheap and
doesn't require API keys or a network connection. The Chainlit UI and the FastAPI
``/chat`` endpoint both consume the graph built here, so the logic lives in one place.
"""

import aiosqlite

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph_supervisor import create_supervisor

from langchain.messages import HumanMessage, AIMessage

from tube_scholar.core.config import settings
from tube_scholar.core.models import chat_model
from tube_scholar.func import State, IntentClassifier
from tube_scholar.video_ingestion import VideoIngestionPipeline

TEMP = 0.7
MAX_ROUNDS_OF_REVISIONS = 2
MIN_VIDS_CONFIDENCE = 3


# --- lazy singletons -------------------------------------------------------
# Built once, on first use, and cached. Keeping construction out of import means
# `import tube_scholar.core.graph` has no side effects (no LLM client, no DB, no keys).

_llm = None
_pipeline = None
_builder = None
_graph = None


def get_llm():
    """The shared chat model, created once.

    Provider-agnostic: the provider is read from the ``MAIN_MODEL`` prefix
    (openai:/anthropic:/google_genai:) and that provider's SDK reads its own API
    key from the environment — so no key is passed explicitly here.
    """
    global _llm  # pylint: disable=global-statement
    if _llm is None:
        _llm = chat_model(settings.main_model, temperature=TEMP)
    return _llm


def get_pipeline() -> VideoIngestionPipeline:
    """The video-ingestion pipeline (Chroma vector store + YouTube clients), created once."""
    global _pipeline  # pylint: disable=global-statement
    if _pipeline is None:
        _pipeline = VideoIngestionPipeline(
            embedding_model=settings.embedding_model,
            youtube_api_key=settings.youtube_api_key,
            chroma_dir=settings.chroma_dir,
        )
    return _pipeline


# --- graph nodes -----------------------------------------------------------

def classify_intent(state: State):
    """Classify the user's intent into state so the graph can route before the supervisor."""
    last_message = state["messages"][-1]
    classifier_llm = get_llm().with_structured_output(IntentClassifier)
    try:
        result = classifier_llm.invoke([HumanMessage(last_message.content)])
        return {"message_intent": result.intent}
    except Exception as e:  # pylint: disable=broad-except
        # on any classifier failure, fall back to the supervisor instead of crashing the turn
        print(f"[classify] intent classification failed: {type(e).__name__}: {e}")
        return {"message_intent": "supervisor"}


def route_after_classify(state: State) -> str:
    """Answer cheap social turns directly; route content questions to the supervisor."""
    if state.get("message_intent") == "chitchat":
        return "chitchat"
    return "supervisor"


def chitchat_responder(state: State):
    """Answer greetings/thanks directly, skipping retrieval and delegation."""
    try:
        reply = get_llm().invoke(state["messages"])
    except Exception as e:  # pylint: disable=broad-except
        print(f"[chitchat] llm call failed: {type(e).__name__}: {e}")
        reply = AIMessage("Sorry, I'm having trouble responding right now. Please try again.")
    return {"messages": [reply]}


def extract_text(message) -> str:
    """Best-effort plain text from a message whose content may be a str or a list of
    content blocks (the model returns blocks like {"type": "text", ...})."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


# --- supervisor prompt -----------------------------------------------------

SUPERVISOR_SYSTEM_MESSAGE = f"""
You are a supervisor coordinating a sub-agent. You delegate by calling it as a tool:
- verification_agent(context): proof-reads candidate output and flags misinformation.

Goal: return a final answer that is both relevant and accurate, in as few delegations as possible.

Routing:
- If the retrieved context is not relevant to the user's query, do not answer from it;
        request better context (or ask the user to narrow the query) instead of fabricating.
- Once you have drafted an answer from relevant context, call verification_agent on it.
  - If it flags misinformation, revise and re-verify (max {MAX_ROUNDS_OF_REVISIONS} rounds),
        then stop.
- Do not call the agent twice for the same input.
    If verification passes, return the answer and stop delegating.

Confidence & attribution:
- Prefer at least {MIN_VIDS_CONFIDENCE} distinct ingested videos backing a claim
  before stating it as an absolute, general truth.
  Judge this from the `source` of your search results (count distinct video_ids)or `list_videos`.
- When fewer than {MIN_VIDS_CONFIDENCE} distinct videos support a claim,
  do NOT state it as settled fact and do NOT tell the user there aren't enough videos ingested.
  Instead, attribute and soften it, frame it as a point of view rather than a verified conclusion,
  e.g. "This is what <creator> says about this topic..." or "Based on <creator>'s take, ...".
- The more independent videos agree, the more confidently you may state something.
  A single source is always an attributed opinion, never a general fact.

Guardrails:
- Never invent facts. If you lack information, delegate or ask, don't guess.
- One delegation at a time; use each agent's result before the next call.

Output:
- Return the final answer, then one line: what you delegated and the outcome.
- If the verification_agent's "SOURCES CHECKED" line names any Wikipedia/Wikidata
  lookups (i.e. it is not "none"), add a short closing note crediting them, e.g.
  "Fact-checked against Wikipedia/Wikidata: Ada Lovelace." Name the entities it checked.
  If SOURCES CHECKED was "none", do NOT add such a note.
- If blocked, ask only for the specific thing you need to proceed.
"""


# --- graph construction ----------------------------------------------------

def _get_builder() -> StateGraph:
    """Build (once) the uncompiled graph: classify -> route -> (chitchat | supervisor) -> END."""
    global _builder  # pylint: disable=global-statement
    if _builder is not None:
        return _builder

    # Imported here (not at module top) so importing this module doesn't construct the
    # verification agent's LLM — keeps app boot cheap and key-free.
    from tube_scholar.agents import verification_agent  # pylint: disable=import-outside-toplevel

    supervisor = create_supervisor(
        agents=[verification_agent],
        tools=get_pipeline().get_tools(),
        model=chat_model(settings.main_model, temperature=TEMP),
        prompt=SUPERVISOR_SYSTEM_MESSAGE,
        output_mode="full_history",
    ).compile()

    builder = StateGraph(State)
    builder.add_node("classify", classify_intent)
    builder.add_node("chitchat", chitchat_responder)
    builder.add_node("supervisor", supervisor)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_after_classify, {
        "chitchat": "chitchat",
        "supervisor": "supervisor",
    })
    builder.add_edge("chitchat", END)
    builder.add_edge("supervisor", END)

    _builder = builder
    return _builder


async def get_graph():
    """Checkpointed graph for normal runs; writes to the persistent SQLite DB.

    Async because AsyncSqliteSaver must be constructed inside the running event loop.
    """
    global _graph  # pylint: disable=global-statement
    if _graph is None:
        conn = await aiosqlite.connect(settings.checkpoint_db)
        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()
        _graph = _get_builder().compile(checkpointer=checkpointer)
    return _graph


def build_eval_graph():
    """Isolated graph for evaluation runs; never writes to the persistent DB."""
    return _get_builder().compile(checkpointer=InMemorySaver())


# --- streaming helpers (shared by Chainlit + the /chat API) ----------------

# Nodes whose LLM output is meant for the user: "chitchat" (this graph's social-reply
# node) and "agent" (the react node inside the supervisor that produces its final answer).
# The verification_agent also runs under an "agent" node but is filtered by namespace.
ANSWER_NODES = {"chitchat", "agent"}


def is_internal_namespace(namespace) -> bool:
    """Tokens from the verification_agent subgraph are internal proof-reading, not answer text."""
    return any("verification_agent" in part for part in namespace)


def is_answer_token(chunk, namespace, metadata) -> bool:
    """True when a streamed chunk is user-facing answer text (not internal agent chatter)."""
    from langchain.messages import AIMessageChunk  # pylint: disable=import-outside-toplevel
    return (
        isinstance(chunk, AIMessageChunk)
        and metadata.get("langgraph_node") in ANSWER_NODES
        and not is_internal_namespace(namespace)
    )
