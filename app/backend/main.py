'''The Python file where the main backend code is located'''


from dotenv import load_dotenv
# import pandas as pd
import os, uuid

from langchain.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model

from videoIngestion import VideoIngestionPipeline
from func import State, IntentClassifier
from app.backend.sub_agents import verification_agent

load_dotenv()
UTUBE_API = os.getenv("YOUTUBE_API_KEY")
GEMINI_API = os.getenv("PAID_GEMINI_API") # PAID_GEMINI_API / FREE_GEMINI_API

temp = .7
llm = init_chat_model(model="google_genai:gemini-3.1-flash-lite", api_key=GEMINI_API, temperature=temp)
pipeline = VideoIngestionPipeline(google_api_key=UTUBE_API)
tools = pipeline.get_tools()
tooled_llm = llm.bind_tools(tools)
checkpointer = InMemorySaver()
max_rounds_of_revisions = 2

def classify_intent(state: State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(IntentClassifier)

    result = classifier_llm.invoke([
        HumanMessage(last_message.content)
    ])

    return {"message_intent": result.intent}




# --- Supervisor Agent ------------------------------------------------------

supervisor_system_message = f"""
You are a supervisor coordinating two sub-agents. You delegate by calling them as tools:
- verification_agent(context): proof-reads candidate output and flags misinformation.
- relevance_grader_agent(context, query): scores how relevant the context is to the user query.

Goal: return a final answer that is both relevant and accurate, in as few delegations as possible.

Routing:
- First call relevance_grader_agent on the retrieved context.
  - If relevance is low, do not answer from it — request better context (or ask the user to narrow the query) instead of fabricating.
- Once context is relevant, call verification_agent on the drafted answer.
  - If it flags misinformation, revise and re-verify (max {max_rounds_of_revisions} rounds), then stop.
- Do not call an agent twice for the same input. If both checks pass, return the answer and stop delegating.

Guardrails:
- Never invent facts. If you lack information, delegate or ask — don't guess.
- One delegation at a time; use each agent's result before the next call.

Output:
- Return the final answer, then one line: what you delegated and the outcome.
- If blocked, ask only for the specific thing you need to proceed.
"""

# ---------------------------------------------------------------------------




# TODO create enviroment identifier ['single_video', 'topic_search', 'personal_collection']

messages = [
    SystemMessage(),   # role/instructions - always index 0
    HumanMessage(),    # the user query
    AIMessage(),       # the model's tool-call request (has .tool_calls)
    ToolMessage(),     # tool output - must reference that call's id
]

if __name__ == '__main__':

    supervisor = create_supervisor(
        agents=[verification_agent],
        tools=[],
        model=init_chat_model(model="google_genai:gemini-3.1-flash-lite", api_key=GEMINI_API, temperature=temp),
        prompt=supervisor_system_message,

        output_mode='full_history'
    ).compile(checkpointer)
    
    # --- query -> intention_agent -> tools (if needed) -> retrives full context -> parses context to text agent -------------

    mes = [
        "Hey, howz it goin",
        "what vids are we workin with",
        "but didnt u say that we werent going to talk about this case anymore",
        "can you get the other videos then",
        "ahah!",
    ]

    for me in mes:
        state = {"messages": [HumanMessage(me)]}
        print(classify_intent(state))

