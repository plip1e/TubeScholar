'''The python file that is home to all the project's agents'''

from dotenv import load_dotenv
import os, uuid

from langgraph.graph import StateGraph, START, END
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from func import State

load_dotenv()
UTUBE_API = os.getenv("YOUTUBE_API_KEY")
GEMINI_API = os.getenv("PAID_GEMINI_API") # PAID_GEMINI_API / FREE_GEMINI_API

max_rounds_of_revisions = 2
temp = .7

# --- Verifictaion Agent -------------------------------------------------------------

def verification_agent():
    pass

# ---------------------------------------------------------------------------


