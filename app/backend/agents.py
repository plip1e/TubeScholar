'''The python file that is home to all the project's agents'''

from dotenv import load_dotenv
import os, uuid

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from func import State, WikiVerifier

load_dotenv()
UTUBE_API = os.getenv("YOUTUBE_API_KEY")
GEMINI_API = os.getenv("PAID_GEMINI_API") # PAID_GEMINI_API / FREE_GEMINI_API

temp = .7

llm = init_chat_model(
    model="google_genai:gemini-3.1-flash-lite",
    api_key=GEMINI_API,
    temperature=temp,
)

# --- Verification Agent -------------------------------------------------------------

verification_system_message = """
You proof-read a candidate answer against the context it was drawn from and flag misinformation.

Given the drafted answer (and any context provided), check each claim:
- Is it supported by the context, or is it unsupported / contradicted?
- Are there overstatements, invented specifics, or hallucinated sources?

Respond with:
- VERDICT: pass | revise
- A short bullet list of any flagged claims and why. If nothing is wrong, say so.
Do not rewrite the answer yourself — just flag. Be concise.

When a claim concerns a real person (e.g. their occupation, field, or education),
verify it against Wikidata: use `search_person` to find the right entity when the
name is ambiguous, then `get_profile` (or `get_profile_by_qid`) to pull their
occupations / field_of_work / education. These come back as Wikidata QIDs — exact,
language-independent match keys — so compare QIDs rather than strings. Use
`get_property` for other facts (birth date, nationality, awards), and `humanise_qid`
only when you need a readable label to phrase a flagged item.

For non-person claims (events, places, concepts, organisations), use `wiki_search`
to pull the relevant Wikipedia article intro and check the claim against it.
"""

wiki = WikiVerifier()

# A react agent with no tools is fine here — verification is pure reasoning over
# the text it's handed. `name` is what the supervisor routes on (and what
# create_supervisor reads via `agent.name`).
verification_agent = create_agent(
    model=llm,
    tools=list(wiki.get_tools().values()),
    system_prompt=verification_system_message,
    name="verification_agent",
)

# ---------------------------------------------------------------------------
