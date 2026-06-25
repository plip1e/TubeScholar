import os, time
import uuid
import requests
import pandas as pd
from pathlib import Path
from langchain.chat_models import init_chat_model

from langgraph.graph.message import add_messages
from langchain.tools import tool
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field

# --- Core ------------------------------------------------------------------

class IntentClassifier(BaseModel):
    reasoning: str = Field(
        description="One sentence: why this turn falls into the chosen intent."
    )
    intent: Literal['new_request', 'follow_up', 'corpus_action', 'meta', 'chitchat'] = Field(
        description=(
            "new_request: a question about video content not yet asked. "
            "  e.g. 'what bosses does he skip?' "
            "follow_up: references the previous answer; needs history to resolve referents. "
            "  e.g. 'what about the second one?' — even if it adds a new sub-topic. "
            "meta: about the assistant itself — its coverage, sources, or how it scores trust. "
            "  e.g. 'which videos do you have?', 'how do you decide a channel is credible?' "
            "chitchat: greeting, thanks, or social with no information need. e.g. 'hey', 'thanks!'"
            "corpus_action: user is asking to change what content is loaded — fetch, add, switch collection"
            " e.g. 'can you get [x] video', 'can we change to [x] topic', 'can you forget about [x] video' "
        )
    )
    needs_clarification: bool = Field(
        default=False,
        description="True if the query is too vague to retrieve on without guessing a referent."
    )

class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_intent: str | None
    context: str | None

class VideoList:
    """A single ingested video's quick-access metadata.

    The pipeline keeps one of these per video in an in-memory registry so an
    agent tool can read a creator / title / stats directly by list placement,
    instead of running a similarity search over the whole vector store.
    """

    def __init__(self, lst_placement, video_id, url, creator, creator_description,
                 title=None, channel_id=None, published_at=None,
                 view_count=None, like_count=None, duration=None):
        self.lst_placement = lst_placement  # incremental, 1-based
        self.video_id = video_id
        self.url = url
        self.video_creator = creator
        self.video_creator_description = creator_description
        self.title = title
        self.channel_id = channel_id
        self.published_at = published_at
        self.view_count = view_count
        self.like_count = like_count
        self.duration = duration

    def as_dict(self) -> dict:
        """Flat dict for an agent tool to return as its result."""
        return {
            "lst_placement": self.lst_placement,
            "video_id": self.video_id,
            "url": self.url,
            "creator": self.video_creator,
            "creator_description": self.video_creator_description,
            "title": self.title,
            "channel_id": self.channel_id,
            "published_at": self.published_at,
            "view_count": self.view_count,
            "like_count": self.like_count,
            "duration": self.duration,
        }

# ---------------------------------------------------------------------------


class RAGEvaluator:
    """
    Purely local RAG evaluator using Gemini-as-judge.
    No RAGAS, no LangSmith dataset — just CSV output + DataFrame.

    Metrics (all scored 0.0 – 1.0 by Gemini):
        Precision    – are the retrieved chunks relevant to the question?
        Recall       – does the context cover the reference answer?
        Faithfulness – is the answer grounded in the context (no hallucinations)?
        Relevance    – does the answer actually address the question?
    """

    RESULTS_DIR    = Path("../data/LS-results")
    QUESTIONS_PATH = Path("../data/LS-questions.csv")

    def __init__(self, graph, pipeline, api_key: str):
        """
        Args:
            graph:    compiled LangGraph graph
            pipeline: VideoIngestionPipeline (for vectorstore access)
            api_key:  Gemini API key — uses gemini-2.5-flash-lite as judge
        """
        self.graph    = graph
        self.pipeline = pipeline

        self.llm = init_chat_model(
            model="google_genai:gemini-3.1-flash-lite",
            api_key=api_key,
            temperature=0
        )

        self.examples = self._load_examples()
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _load_examples(self) -> list[dict]:
        df = pd.read_csv(self.QUESTIONS_PATH)
        return df[["question", "answer"]].to_dict(orient="records")

    def _next_version(self) -> str:
        existing = list(self.RESULTS_DIR.glob("rag-v*.csv"))
        if not existing:
            return "rag-v1"
        nums = []
        for f in existing:
            try:
                nums.append(int(f.stem.replace("rag-v", "")))
            except ValueError:
                pass
        return f"rag-v{max(nums) + 1}"

    def _rag_pipeline(self, question: str) -> dict:
        """Run graph + pull context chunks from vectorstore."""
        thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        docs     = self.pipeline.vectorstore.similarity_search(question, k=3)
        contexts = [doc.page_content for doc in docs]
        result   = self.graph.invoke(
            {"messages": [{"role": "user", "content": question}]},
            thread_config
        )
        return {"answer": result["messages"][-1].content, "contexts": contexts}

    def _score(self, prompt: str) -> float:
        """Send a scoring prompt to Gemini, parse back a 0-1 float."""
        response = self.llm.invoke([
            {
                "role": "system",
                "content": (
                    "You are a strict evaluation assistant. "
                    "Respond with ONLY a single float between 0.0 and 1.0. "
                    "No explanation, no extra text — just the number."
                )
            },
            {"role": "user", "content": prompt}
        ])
        # print(response)
        # raise KeyError
        try:
            return round(float(response.content[-1]["text"].strip()), 3)
        except ValueError:
            return 0.0

    # --- individual metrics -------------------------------------------

    def _precision(self, question: str, contexts: list[str]) -> float:
        ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        return self._score(f"""Score how relevant the retrieved context chunks are to the question.

Question: {question}

Retrieved Contexts:
{ctx}

1.0 = all chunks are highly relevant to the question
0.5 = some chunks are relevant, some are off-topic
0.0 = none of the chunks are relevant to the question""")

    def _recall(self, question: str, contexts: list[str], reference: str) -> float:
        ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        return self._score(f"""Score whether the retrieved context contains enough information to produce the reference answer.

Question: {question}
Reference Answer: {reference}

Retrieved Contexts:
{ctx}

1.0 = context fully covers everything needed for the reference answer
0.5 = context partially covers it, some key info is missing
0.0 = context is missing most or all of the key information""")

    def _faithfulness(self, answer: str, contexts: list[str]) -> float:
        ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        return self._score(f"""Score whether every claim in the answer is supported by the retrieved context.

Answer: {answer}

Retrieved Contexts:
{ctx}

1.0 = every claim in the answer is directly supported by the context
0.5 = most claims are supported but some appear to be inferred or hallucinated
0.0 = the answer contains claims that contradict or are absent from the context""")

    def _relevance(self, question: str, answer: str) -> float:
        return self._score(f"""Score how directly and completely the answer addresses the question.

Question: {question}
Answer: {answer}

1.0 = answer directly and fully addresses the question
0.5 = answer partially addresses the question or drifts off-topic
0.0 = answer does not address the question at all""")

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def get_tools(self) -> dict:

        @tool
        def precision(question: str, contexts: list[str]) -> float:

            return self._precision(question, contexts)
        
        @tool
        def recall(question: str, contexts: list[str], reference: str) -> float:

            return self._recall(question, contexts, reference)
        
        @tool
        def faithfulness(answer: str, contexts: list[str]) -> float:

            return self._faithfulness(answer, contexts)
        
        @tool
        def relevance(question: str, answer: list[str]) -> float:

            return self._relevance(question, answer)

        return {
            'precision': precision,
            'recall': recall,
            'faithfulness': faithfulness,
            'relevance': relevance
        }

    def run(self) -> pd.DataFrame:
        """
        Evaluate all examples, save results to CSV, return DataFrame.
        Rows: one per question + one AVERAGE summary row at the bottom.
        """
        version = self._next_version()
        print(f"Running experiment: {version}\n")

        rows = []
        for i, ex in enumerate(self.examples):
            print(f"[{i+1}/{len(self.examples)}] {ex['question'][:60]}...")

            out = self._rag_pipeline(ex["question"])

            rows.append({
                "Question":    ex["question"],
                "Precision":   self._precision(ex["question"], out["contexts"]),
                "Recall":      self._recall(ex["question"], out["contexts"], ex["answer"]),
                "Faithfulness":self._faithfulness(out["answer"], out["contexts"]),
                "Relevance":   self._relevance(ex["question"], out["answer"]),
            })
            print(f"""
                       P={rows[-1]['Precision']}  R={rows[-1]['Recall']}  
                       F={rows[-1]['Faithfulness']}  Rel={rows[-1]['Relevance']}""")
            
            with open('res.txt', 'a') as a:
                a.write(f"CONTEXTS:, {out["contexts"]}\nANSWER:, {out["answer"]}\n\n")

        df = pd.DataFrame(rows)

        # summary row
        avg_row = pd.DataFrame([{
            "Question":    "AVERAGE",
            "Precision":   round(df["Precision"].mean(),    3),
            "Recall":      round(df["Recall"].mean(),       3),
            "Faithfulness":round(df["Faithfulness"].mean(), 3),
            "Relevance":   round(df["Relevance"].mean(),    3),
        }])
        df = pd.concat([df, avg_row], ignore_index=True)

        save_path = self.RESULTS_DIR / f"{version}.csv"
        df.to_csv(save_path, index=False)
        print(f"\nSaved → {save_path}")

        return df

# --- wiki ------------------------------------------------------------------


class WikiVerifier:
    """Wikidata lookup helper for the verification agent.

    Resolves a person's name to a Wikidata entity (QID) and pulls structured
    facts — occupations, field of work, and education — that an answer can be
    fact-checked against.

    QIDs (e.g. "Q82955") are kept as the primary return value: they are a
    language-independent, exact match key. The agent compares the QIDs found in
    a person's claims against reference QIDs rather than fuzzy-matching strings.
    `humanise_qid` is provided for when the LLM needs a readable label to put in
    a prompt or answer.
    """

    WIKIDATA_API = "https://www.wikidata.org/w/api.php"
    WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
    HEADERS = {"User-Agent": "TubeScholar/0.1 (ak47dodger@gmail.com)"}

    # human-friendly name -> Wikidata property id
    PROPS = {
        "occupations": "P106",
        "field_of_work": "P101",
        "major": "P812",
        "degree": "P512",
        "educated_at": "P69",
    }

    # ------------------------------------------------------------------
    # internals (requests-only; ported from test.ipynb)
    # ------------------------------------------------------------------

    def _find_qid(self, name: str, lang: str = "en") -> list[dict]:
        """Search Wikidata entities by name -> candidate {id, label, description}."""
        r = requests.get(
            self.WIKIDATA_API,
            params={
                "action": "wbsearchentities",
                "search": name,
                "language": lang,
                "format": "json",
                "limit": 5,
            },
            headers=self.HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        return [
            {
                "id": res["id"],
                "label": res.get("label"),
                "description": res.get("description"),
            }
            for res in r.json()["search"]
        ]

    def _get_claims(self, qid: str) -> dict:
        """Pull only the claims block for an entity."""
        r = requests.get(
            self.WIKIDATA_API,
            params={
                "action": "wbgetentities",
                "ids": qid,
                "props": "claims",
                "format": "json",
            },
            headers=self.HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["entities"][qid]["claims"]

    def _extract_property_qids(self, claims: dict, prop: str) -> list[str]:
        """All entity-valued QIDs for a property; missing property -> []."""
        out = []
        for stmt in claims.get(prop, []):
            snak = stmt.get("mainsnak", {})
            if snak.get("snaktype") != "value":  # skip novalue/somevalue
                continue
            qid = snak.get("datavalue", {}).get("value", {})
            if isinstance(qid, dict) and qid.get("id"):
                out.append(qid["id"])
        return out

    def _labels_for(self, qids: list[str], lang: str = "en") -> dict:
        """Batch-resolve QIDs -> {qid: label}. Wikidata caps ids at 50/call."""
        labels = {}
        for i in range(0, len(qids), 50):
            batch = qids[i:i + 50]
            r = requests.get(
                self.WIKIDATA_API,
                params={
                    "action": "wbgetentities",
                    "ids": "|".join(batch),
                    "props": "labels",
                    "languages": lang,
                    "format": "json",
                },
                headers=self.HEADERS,
                timeout=10,
            )
            r.raise_for_status()
            for qid, ent in r.json().get("entities", {}).items():
                lbl = ent.get("labels", {}).get(lang, {}).get("value")
                if lbl:
                    labels[qid] = lbl
        return labels

    def _profile_from_claims(self, name: str, qid: str, claims: dict) -> dict:
        return {
            "name": name,
            "qid": qid,
            "url": f"https://www.wikidata.org/wiki/{qid}",
            "occupations": self._extract_property_qids(claims, self.PROPS["occupations"]),
            "field_of_work": self._extract_property_qids(claims, self.PROPS["field_of_work"]),
            "education": {
                "major": self._extract_property_qids(claims, self.PROPS["major"]),
                "degree": self._extract_property_qids(claims, self.PROPS["degree"]),
                "educated_at": self._extract_property_qids(claims, self.PROPS["educated_at"]),
            },
        }

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def get_profile(self, name: str) -> dict:
        """Resolve a name to its best Wikidata match and return its profile."""
        candidates = self._find_qid(name)
        if not candidates:
            return {"error": f"no Wikidata match for {name!r}"}
        qid = candidates[0]["id"]
        return self._profile_from_claims(name, qid, self._get_claims(qid))

    def get_profile_by_qid(self, qid: str) -> dict:
        """Return the profile for a known QID (skips name search)."""
        return self._profile_from_claims(qid, qid, self._get_claims(qid))

    def search_person(self, name: str) -> list[dict]:
        """Return candidate matches (id, label, description) for disambiguation."""
        return self._find_qid(name)

    def get_property(self, qid: str, prop: str) -> list:
        """Return all values for any Wikidata property of an entity.

        Entity-valued statements come back as QIDs; literal statements (dates,
        strings, quantities) come back as their raw datavalue.
        """
        claims = self._get_claims(qid)
        out = []
        for stmt in claims.get(prop, []):
            snak = stmt.get("mainsnak", {})
            if snak.get("snaktype") != "value":
                continue
            value = snak.get("datavalue", {}).get("value")
            if isinstance(value, dict) and value.get("id"):
                out.append(value["id"])  # entity reference
            else:
                out.append(value)  # literal (time/string/quantity/...)
        return out

    def humanise_qid(self, qid: str) -> str:
        """Resolve a single QID to its English label (or the QID if unresolved)."""
        return self._labels_for([qid]).get(qid, qid)

    def wiki_search(self, query: str, results: int = 3) -> list[dict]:
        """General Wikipedia lookup for any topic (not just people).

        Returns the best-matching articles as {title, extract, url}, where
        extract is the plain-text intro of the article.
        """
        r = requests.get(
            self.WIKIPEDIA_API,
            params={
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": results,
                "prop": "extracts|info",
                "exintro": 1,
                "explaintext": 1,
                "inprop": "url",
                "redirects": 1,
            },
            headers=self.HEADERS,
            timeout=10,
        )
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        ordered = sorted(pages.values(), key=lambda p: p.get("index", 0))
        return [
            {
                "title": p.get("title"),
                "extract": (p.get("extract") or "").strip(),
                "url": p.get("fullurl"),
            }
            for p in ordered
        ]

    def get_tools(self) -> dict:

        @tool
        def get_profile(name: str) -> dict:
            """Look up a person on Wikidata by name and return their profile:
            occupations (P106), field_of_work (P101) and education (major/degree/
            educated_at) as Wikidata QIDs, plus the entity QID and wiki URL.
            QIDs are exact, language-independent match keys — compare them against
            reference QIDs. Returns {"error": ...} if no match is found."""
            return self.get_profile(name)

        @tool
        def get_profile_by_qid(qid: str) -> dict:
            """Same profile as get_profile, but for a QID you already have
            (e.g. one chosen via search_person). Skips the name search."""
            return self.get_profile_by_qid(qid)

        @tool
        def search_person(name: str) -> list:
            """Search Wikidata for people matching a name. Returns up to 5
            candidates as {id, label, description} so you can disambiguate name
            collisions before fetching a profile."""
            return self.search_person(name)

        @tool
        def get_property(qid: str, prop: str) -> list:
            """Fetch any Wikidata property of an entity by P-code (e.g. P569 date
            of birth, P27 country of citizenship, P166 awards). Entity values are
            returned as QIDs; literals (dates/strings/quantities) as raw values."""
            return self.get_property(qid, prop)

        @tool
        def humanise_qid(qid: str) -> str:
            """Convert a single Wikidata QID into its English label, for when you
            need a human-readable term in your answer (e.g. Q82955 -> 'politician')."""
            return self.humanise_qid(qid)

        @tool
        def wiki_search(query: str) -> list:
            """General Wikipedia search for ANY topic (events, places, concepts,
            organisations — not just people). Returns the top matching articles as
            {title, extract, url}, where extract is the article's plain-text intro.
            Use this to fact-check non-person claims."""
            return self.wiki_search(query)

        return {
            "get_profile": get_profile,
            "get_profile_by_qid": get_profile_by_qid,
            "search_person": search_person,
            "get_property": get_property,
            "humanise_qid": humanise_qid,
            "wiki_search": wiki_search,
        }


# ---------------------------------------------------------------------------

