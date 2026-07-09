# 🎓 TubeScholar

> A trustworthy research assistant for YouTube, not just another summarizer.

TubeScholar turns YouTube videos into a **queryable, source-aware knowledge base**. Ask a question, get an answer grounded in the actual transcript, *plus* a transparent trust signal that tells you how much you should rely on the source it came from.

The name is intentional: **Tube** (YouTube) + **Scholar** (Google Scholar), a serious research and education tool, not a TL;DR machine.

---

## The Problem

YouTube is one of the largest knowledge bases on the planet, but it is also unverified. Plain summarizers will happily condense a confident, wrong video into a clean paragraph and hand it to you with zero context about whether the person talking knows what they are saying.

TubeScholar's whole reason to exist is the layer most tools skip: **should you trust this answer, and why?**

---

## ⭐ The Trust Layer

Every answer is meant to carry a trust assessment, not just text. The trust layer is built from three signals:

- **Claim verification**: are the specific claims supported by the retrieved context and by external reference sources?
- **Attribution and confidence**: how many independent sources back a claim, and is it a single creator's opinion or a broadly supported fact?
- **Channel reputation**: the speaker's relevant authority and the channel's track record.

These are surfaced **transparently** alongside the response, so the user sees the reasoning, not just a number.

### What is built today

- **A verification agent** that proof-reads each drafted answer and fact-checks claims against **Wikidata** and **Wikipedia**. For claims about a real person (occupation, field of work, education) it resolves the name to a Wikidata entity (QID) and compares structured facts; for events, places, and concepts it checks against the relevant Wikipedia article. It flags unsupported or contradicted claims with a `pass` / `revise` verdict rather than rewriting the answer, and it now reports exactly which sources it consulted so the supervisor can credit them.
- **Wiki attribution in the answer**: when the verification agent actually used a Wikipedia or Wikidata lookup, the supervisor appends a short "Fact-checked against Wikipedia/Wikidata" note (naming the entities checked) to the final answer. When no wiki tool was used, no note is added.
- **Attribution and confidence heuristics** enforced by the supervisor: a single source is always framed as that creator's point of view, never settled fact, and a claim is only stated as a general truth when several distinct ingested videos agree (target: 3 or more).

### Planned

- **Channel reputation scoring** (track record, reliability signals vs red flags) and a dedicated relevance-grading agent.
- A **blended trust score** that combines verification, attribution, and reputation into one transparent, explained signal.

---

## 🧭 Usage Modes

| Mode | What it does | Status |
|------|--------------|--------|
| **Personal Collection** | Build and query your own curated corpus of videos *(core scope)*. | Built |
| **Single Video** | Deep Q&A against one video's transcript. | Planned |
| **Topic Search** | Search YouTube for a topic, pull relevant videos, answer across them. | Planned |

Retrieval can already be scoped to a single `video_id`, and the corpus is fully user-managed (add, list, remove), so the building blocks for all three modes exist. An explicit mode selector is still to come.

---

## 🏗️ Architecture

TubeScholar is a **multi-agent system** built on a LangGraph `StateGraph`. An intent classifier routes each turn before it reaches the supervisor, so cheap social turns skip retrieval entirely. The supervisor then coordinates sub-agents (retrieval, verification, relevance grading, trust scoring) instead of running everything through a single monolithic chain.

The same graph is driven by two front ends: a **React (TypeScript)** web app talking to a **FastAPI** backend that streams answers over Server-Sent Events, and a **Chainlit** chat UI kept around as a quick debug surface. The graph is built lazily, so importing the app has no side effects and needs no API keys.

This diagram shows the finished shape of the system (planned nodes included):

```
User query
   |
   v
[ Classify ]  intent: new_request / follow_up / corpus_action / meta / chitchat
   |
   |--> Chitchat node --> direct reply (no retrieval)
   |
   '--> Supervisor
           |
           |--> Retrieval tools     --> ChromaDB (vector search over transcripts)
           |--> Corpus tools        --> ingest / search YouTube / list / delete videos
           |--> Relevance grader    --> is the context good enough to answer from?  (planned)
           |--> Verification agent  --> Wikidata + Wikipedia fact-check
           '--> Trust scorer        --> reputation + attribution + confidence  (planned)
                    |
                    v
           Grounded, trust-annotated answer --> FastAPI (SSE) --> React UI / Chainlit
```

Transcripts are ingested via `youtube-transcript-api` (through rotating proxies), chunked, embedded, and persisted in ChromaDB. When a video has no captions, the pipeline falls back to a metadata and top-comments overview (clearly labelled as inferred). An `openai-whisper` speech-to-text path exists but is gated off by default while audio transcription is offloaded to an external service.

---

## 🔌 Provider-agnostic models

Chat models and embeddings are not tied to any single vendor. Both are selected by a `provider:model` string, so you can run TubeScholar on **Google (Gemini)**, **OpenAI**, or **Anthropic (Claude)** by editing two environment variables and setting the matching key.

- `MAIN_MODEL` picks the chat model, for example `google_genai:gemini-3.1-flash-lite`, `openai:gpt-4o`, or `anthropic:claude-sonnet-4-5`.
- `EMBEDDING_MODEL` picks the embedding model, for example `google_genai:models/gemini-embedding-001` or `openai:text-embedding-3-small`.
- Each provider reads its own standard key from the environment (`GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`), so chat and embeddings can even use different providers.

Google ships as a bundled dependency; OpenAI and Anthropic are optional extras (see Setup). Two notes worth knowing:

- Anthropic has no embeddings API, so a Claude chat setup must use another provider for `EMBEDDING_MODEL` (for example OpenAI).
- Each embedding model gets its own ChromaDB collection (vector sizes differ between models), so switching `EMBEDDING_MODEL` means re-ingesting your videos once under the new model.

---

## ✅ Current Progress

- **Installable package and API**: the backend is a src-layout Python package (`tube_scholar`) installed in editable mode. A **FastAPI** backend exposes `/health`, a token-streaming `/chat` endpoint, `/ingest` (video ingestion over HTTP), and `/videos` (the ingested library) — sharing the exact same LangGraph and pipeline as every other surface.
- **Web UI**: a **Vite + React + TypeScript** single-page app with token-by-token streaming chat (`fetch` + `ReadableStream` over SSE) and a collapsible video-library panel with URL ingestion. The panel auto-refreshes after each answer, so videos the agent ingests mid-conversation appear immediately.
- **Provider-agnostic LLM stack**: chat via `init_chat_model` and embeddings via `init_embeddings`, chosen by a `provider:model` prefix, with per-provider API keys and a clear error message when a key is missing.
- **Ingestion pipeline**: YouTube Data API metadata, caption retrieval through Webshare rotating proxies, concurrent multi-URL ingestion on a bounded thread pool with per-thread HTTP clients (thread-safe), word-window chunking, embeddings, and persistent ChromaDB storage. Deterministic chunk IDs give idempotent re-ingestion, a 7-day staleness check skips up-to-date videos, and an in-memory registry is rebuilt from the store on startup.
- **Graceful failure handling**: every tool returns an agent-readable status dict (invalid URL, not found, quota/blocked, transcript unavailable, empty transcript) instead of crashing the run, including a metadata and top-comments fallback when no transcript exists.
- **Agent graph**: structured intent classification, a direct chitchat path, and a `langgraph_supervisor` supervisor wired with the full pipeline toolset plus the verification sub-agent. Conversation history is token-capped per thread and persisted via an async SQLite checkpointer.
- **Verification agent**: a tool-using agent backed by a Wikidata/Wikipedia client (`get_profile`, `search_person`, `get_property`, `humanise_qid`, `wiki_search`) that reports the sources it checked back to the supervisor.
- **Answer-only streaming**: subgraph-aware token filtering shared by both front ends, so only user-facing answer tokens are streamed (internal proof-reading stays hidden).
- **Evaluation harness**: a local LLM-as-judge `RAGEvaluator` scoring precision, recall, faithfulness, and relevance, with versioned CSV output and LangSmith tracing.

---

## 🧰 Tech Stack

| Layer | Choice |
|-------|--------|
| **Orchestration** | LangGraph + LangChain (`langgraph_supervisor`) |
| **Chat model** | Provider-agnostic via `init_chat_model` (Google, OpenAI, or Anthropic). Default: `google_genai:gemini-3.1-flash-lite` |
| **Embeddings** | Provider-agnostic via `init_embeddings`. Default: `google_genai:models/gemini-embedding-001` |
| **Vector store** | ChromaDB (persistent, one collection per embedding model) |
| **Backend API** | FastAPI with Server-Sent Events streaming |
| **Transcripts** | `youtube-transcript-api` + `openai-whisper` (CPU fallback, currently gated off) |
| **Video metadata / search** | YouTube Data API v3 |
| **External fact-checking** | Wikidata + Wikipedia APIs |
| **Evaluation and tracing** | LangSmith tracing + custom LLM-as-judge `RAGEvaluator` |
| **Frontend** | Vite + React + TypeScript SPA (streaming chat + video library); Chainlit kept as a debug UI |
| **Proxies** | Webshare (rotating, to avoid IP bans during ingestion) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+ (for the web frontend)
- An API key for your chosen provider (default is Google Gemini via [Google AI Studio](https://aistudio.google.com/), free tier is fine for development). For OpenAI or Anthropic, use their key instead.
- A YouTube Data API v3 key
- (Optional) Webshare credentials for rotating proxies
- (Optional) A LangSmith API key for tracing and evaluation

### Setup

```bash
# clone
git clone https://github.com/plip1e/TubeScholar.git
cd TubeScholar

# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install the package (Google/Gemini is bundled by default)
pip install -e .
```

Optional installs, depending on what you need:

```bash
pip install -e ".[openai]"       # enables openai:* chat and embeddings
pip install -e ".[anthropic]"    # enables anthropic:* chat
pip install -e ".[all]"          # both OpenAI and Anthropic
pip install -e ".[dev]"          # tests, notebooks, and tracing tools
```

Create a `.env` file in the project root (see `.env.example` for the full list):

```env
# Models are "provider:model" strings (provider is google_genai, openai, or anthropic)
MAIN_MODEL=google_genai:gemini-3.1-flash-lite
EMBEDDING_MODEL=google_genai:models/gemini-embedding-001

# Set only the key(s) for the provider(s) you actually use
GOOGLE_API_KEY=your_gemini_key
# OPENAI_API_KEY=your_openai_key
# ANTHROPIC_API_KEY=your_anthropic_key

YOUTUBE_API_KEY=your_youtube_data_api_key

# optional
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=false
WEBSHARE_PROXY_USERNAME=
WEBSHARE_PROXY_PASSWORD=
WHISPER_ENABLED=0                # set to 1 to enable the local Whisper caption fallback
# CHROMA_DIR=data/chroma_db
# CHECKPOINT_DB=data/checkpoints.sqlite
```

### Run (web app: FastAPI + React)

This is the main front end. Start the backend from the project root:

```bash
tubescholar
# or, equivalently:
uvicorn tube_scholar.main:app --reload --app-dir backend/src
```

Then start the frontend dev server:

```bash
cd frontend
npm install          # first time only
npm run dev
```

Open `http://localhost:5173` — a streaming chat with a collapsible video-library panel (top right). Vite proxies `/api/*` to the backend on `:8000`, so no CORS setup is needed in dev. See [frontend/README.md](frontend/README.md) for details.

Interactive API docs are auto-generated at `http://localhost:8000/docs`.

| Endpoint | Method | Status | Description |
|----------|--------|--------|-------------|
| `/health` | GET | Working | Liveness probe. Returns status and version. |
| `/chat` | POST | Working | Streams the answer token by token over SSE. Body: `{"message": "...", "thread_id": "..."}`. |
| `/ingest` | POST | Working | Ingests a YouTube video over HTTP. Body: `{"url": "..."}`. Returns the pipeline's status dict. |
| `/videos` | GET | Working | Lists every ingested video's metadata, ordered by placement. |

### Run (Chainlit debug UI)

The original prototype UI, kept as a quick debug surface over the same graph. Run it *instead of* the FastAPI backend (both default to port 8000):

```bash
chainlit run backend/src/tube_scholar/chainlit_app.py
```

Then open the printed `http://localhost:8000` link in your browser. Avoid the `-w` watch flag here: the SQLite checkpointer writes to `data/` on every turn, and the watcher would treat that as a code change and reload the app mid-request. Only use `-w` while actively editing the app's code.

---

## 🗂️ Project Layout

```
TubeScholar/
  pyproject.toml            package metadata, dependencies, and provider extras
  .env.example              documented environment variables
  backend/
    src/tube_scholar/       the importable Python package
      main.py               FastAPI app (mounts routers, exposes /health)
      chainlit_app.py       Chainlit chat UI (streams the graph)
      core/
        config.py           typed settings loaded from .env
        graph.py            LangGraph assembly (classify -> route -> supervisor)
        models.py           provider-agnostic chat-model construction
      api/
        chat.py             POST /chat  (SSE streaming)
        ingest.py           POST /ingest (video ingestion over HTTP)
        videos.py           GET /videos (the ingested library)
      agents.py             the verification agent
      func.py               state, intent classifier, WikiVerifier, RAGEvaluator
      video_ingestion.py    the YouTube ingestion pipeline
    tests/                  pytest suite
    notebooks/              experiments and scratch work
  frontend/                 Vite + React + TypeScript single-page app
    src/App.tsx             state owner (chat, videos, panel)
    src/api/client.ts       fetch wrappers + SSE stream reader
    src/components/         ChatWindow, MessageBubble, Composer, VideoPanel
  data/                     local vector DB, checkpoints, transcripts (gitignored)
```

Run the test suite with `pytest` (after installing the `dev` extra).

---

## 📊 Evaluation

Evaluation is treated as a **first-class signal**, not an afterthought.

- A custom **LLM-as-judge** `RAGEvaluator` scores **precision, recall, faithfulness, and relevance** (each 0.0 to 1.0), using the same provider-agnostic model configuration as the rest of the app.
- Runs are versioned (`rag-v1`, `rag-v2`, and so on) and written to local CSVs, with one averaged summary row per run. (RAGAS was dropped as incompatible with the modern LangChain stack.)
- The evaluator calls the vector store directly (bypassing the graph) to avoid LangChain format-string errors from curly braces in transcript chunks.
- LangSmith provides tracing across the pipeline via `@traceable`.

---

## 🗺️ Roadmap

**Stretch goals**
- [x] Connect the `/ingest` endpoint and round out the FastAPI surface
- [x] Vite + React frontend over the FastAPI backend
- [ ] Channel reputation scoring and a blended, explained trust score
- [ ] Dedicated relevance-grading agent and explicit usage-mode selector
- [ ] Deployment via Hugging Face Spaces or a VPS
- [ ] Chrome extension

---

## 📄 License

Licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for the full text.
