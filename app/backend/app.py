from dotenv import load_dotenv
import os, uuid
import aiosqlite

from langchain.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage, AIMessageChunk
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model

from videoIngestion import VideoIngestionPipeline
from agents import verification_agent
from func import State, IntentClassifier

import chainlit as cl

load_dotenv()
UTUBE_API = os.getenv("YOUTUBE_API_KEY")
GEMINI_API = os.getenv("PAID_GEMINI_API") # PAID_GEMINI_API / FREE_GEMINI_API

temp = .7
llm = init_chat_model(model="google_genai:gemini-3.1-flash-lite", api_key=GEMINI_API, temperature=temp)
pipeline = VideoIngestionPipeline(google_api_key=GEMINI_API, youtube_api_key=UTUBE_API)
tools = pipeline.get_tools()
tooled_llm = llm.bind_tools(tools)
CHECKPOINT_DB = "../data/checkpoints.sqlite"
max_rounds_of_revisions = 2
min_vids_confidence = 3

def classify_intent(state: State):
    '''Classify the user's intent into state so the graph can route before the supervisor.'''
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(IntentClassifier)

    try:
        result = classifier_llm.invoke([
            HumanMessage(last_message.content)
        ])
        return {"message_intent": result.intent}
    except Exception as e:
        # on any classifier failure, fall back to the supervisor instead of crashing the turn
        print(f"[classify] intent classification failed: {type(e).__name__}: {e}")
        return {"message_intent": "supervisor"}


def route_after_classify(state: State) -> str:
    '''Answer cheap social turns directly; route anything needing video content to the supervisor.'''
    if state.get("message_intent") == "chitchat":
        return "chitchat"
    return "supervisor"


def chitchat_responder(state: State):
    '''Answer greetings/thanks directly, skipping retrieval and delegation.'''
    try:
        reply = llm.invoke(state["messages"])
    except Exception as e:
        print(f"[chitchat] llm call failed: {type(e).__name__}: {e}")
        reply = AIMessage("Sorry, I'm having trouble responding right now. Please try again.")
    return {"messages": [reply]}


def extract_text(message) -> str:
    '''Best-effort plain text from a message whose content may be a str or a list of
    content blocks (Gemini returns blocks like {"type": "text", ...}).'''
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




# TODO create enviroment identifier ['single_video', 'topic_search', 'personal_collection']

# --- Supervisor Agent ------------------------------------------------------

supervisor_system_message = f"""
You are a supervisor coordinating two sub-agents. You delegate by calling them as tools:
- verification_agent(context): proof-reads candidate output and flags misinformation.
- relevance_grader_agent(context, query): scores how relevant the context is to the user query.

Goal: return a final answer that is both relevant and accurate, in as few delegations as possible.

Routing:
- First call relevance_grader_agent on the retrieved context.
  - If relevance is low, do not answer from it, request better context (or ask the user to narrow the query) instead of fabricating.
- Once context is relevant, call verification_agent on the drafted answer.
  - If it flags misinformation, revise and re-verify (max {max_rounds_of_revisions} rounds), then stop.
- Do not call an agent twice for the same input. If both checks pass, return the answer and stop delegating.

Confidence & attribution:
- Prefer at least {min_vids_confidence} distinct ingested videos backing a claim before stating it as an absolute,
  general truth. Judge this from the `source` of your search results (count distinct video_ids)
  or `list_videos`.
- When fewer than {min_vids_confidence} distinct videos support a claim, do NOT state it as settled fact and do NOT
  tell the user there aren't enough videos ingested. Instead, attribute and soften it, frame it
  as a point of view rather than a verified conclusion, e.g.
  "This is what <creator> says about this topic..." or "Based on <creator>'s take, ...".
- The more independent videos agree, the more confidently you may state something. A single source
  is always an attributed opinion, never a general fact.

Guardrails:
- Never invent facts. If you lack information, delegate or ask, don't guess.
- One delegation at a time; use each agent's result before the next call.

Output:
- Return the final answer, then one line: what you delegated and the outcome.
- If blocked, ask only for the specific thing you need to proceed.
"""

pipeline_tools = pipeline.get_tools()

supervisor = create_supervisor(
    agents=[verification_agent],
    tools=pipeline_tools,
    model=init_chat_model(model="google_genai:gemini-3.1-flash-lite", api_key=GEMINI_API, temperature=temp),
    prompt=supervisor_system_message,
    output_mode='full_history'
).compile()

# ---------------------------------------------------------------------------

# --- graph: classify -> route -> (chitchat | supervisor) -> END --------
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

# Lazily built so AsyncSqliteSaver is constructed inside the running event loop.
_graph = None

async def get_graph():
    global _graph
    if _graph is None:
        conn = await aiosqlite.connect(CHECKPOINT_DB)
        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()
        _graph = builder.compile(checkpointer=checkpointer)
    return _graph


def build_eval_graph():
    # Isolated graph for evaluation runs; never writes to the persistent DB.
    return builder.compile(checkpointer=InMemorySaver())

# Nodes whose LLM output is meant for the user: "chitchat" (this graph's social-reply
# node) and "agent" (the react node inside the supervisor that produces its final answer).
# The verification_agent also runs under an "agent" node but is filtered by namespace below.
ANSWER_NODES = {"chitchat", "agent"}


def is_internal_namespace(namespace) -> bool:
    '''Tokens from the verification_agent subgraph are internal proof-reading, not answer text.'''
    return any("verification_agent" in part for part in namespace)


@cl.on_message
async def on_message(message: cl.Message):
    user = cl.user_session.get("user")
    thread_id = user.identifier if user else cl.context.session.id
    config = {"configurable": {"thread_id": thread_id}}
    graph = await get_graph()
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")

    # subgraphs=True so the supervisor's tokens stream through; each item is
    # (namespace, (message_chunk, metadata)).
    async for namespace, (chunk, metadata) in graph.astream(
        {"messages": [HumanMessage(content=message.content)]},
        stream_mode="messages",
        subgraphs=True,
        config=RunnableConfig(callbacks=[cb], **config),
    ):
        if (
            isinstance(chunk, AIMessageChunk)
            and metadata.get("langgraph_node") in ANSWER_NODES
            and not is_internal_namespace(namespace)
        ):
            token = extract_text(chunk)
            if token:
                await final_answer.stream_token(token)

    await final_answer.send()



if __name__ == '__main__':
    pass
