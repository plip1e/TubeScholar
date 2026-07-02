''' File that defines the agents used in the backend of the application. '''

from langchain.agents import create_agent

from tube_scholar.core.config import settings
from tube_scholar.core.models import chat_model
from tube_scholar.func import WikiVerifier

TEMP = .7

# Provider-agnostic: provider comes from the MAIN_MODEL prefix; the provider SDK
# reads its own key from the environment (loaded via core.config).
llm = chat_model(settings.main_model, temperature=TEMP)

# --- Verification Agent -------------------------------------------------------------

VERIFICATION_SYSTEM_MESSAGE = """
You proof-read a candidate answer against the context it was drawn from and flag misinformation.

Given the drafted answer (and any context provided), check each claim:
- Is it supported by the context, or is it unsupported / contradicted?
- Are there overstatements, invented specifics, or hallucinated sources?

Respond with:
- VERDICT: pass | revise
- A short bullet list of any flagged claims and why. If nothing is wrong, say so.
- SOURCES CHECKED: list every Wikipedia/Wikidata lookup you actually performed as
  "<tool> -> <entity/query>" (e.g. "get_profile -> Ada Lovelace"), or the single word
  "none" if you did not call any wiki tool. Report this honestly based on the tools you
  used, not the tools you could have used.
Do not rewrite the answer yourself, just flag. Be concise.

When a claim concerns a real person (e.g. their occupation, field, or education),
verify it against Wikidata: use `search_person` to find the right entity when the
name is ambiguous, then `get_profile` (or `get_profile_by_qid`) to pull their
occupations / field_of_work / education. These come back as Wikidata QIDs (exact,
language-independent match keys), so compare QIDs rather than strings. Use
`get_property` for other facts (birth date, nationality, awards), and `humanise_qid`
only when you need a readable label to phrase a flagged item.

For non-person claims (events, places, concepts, organisations), use `wiki_search`
to pull the relevant Wikipedia article intro and check the claim against it.
"""

wiki = WikiVerifier()

# `name` is what create_supervisor routes on (read via `agent.name`).
verification_agent = create_agent(
    model=llm,
    tools=list(wiki.get_tools().values()),
    system_prompt=VERIFICATION_SYSTEM_MESSAGE,
    name="verification_agent",
)

# ---------------------------------------------------------------------------
