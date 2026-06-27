# 🎓 TubeScholar

> A trustworthy research assistant for YouTube, not just another summarizer.

TubeScholar turns YouTube videos into a **queryable, source-aware knowledge base**. Ask a question, get an answer grounded in the actual transcript, *plus* a transparent trust signal that tells you how much you should rely on the source it came from.

The name is intentional: **Tube** (YouTube) + **Scholar** (Google Scholar), a serious research and education tool, not a TL;DR machine.

---

## The Problem

YouTube is one of the largest knowledge bases on the planet, but it's also unverified. Plain summarizers will happily condense a confident, wrong video into a clean paragraph and hand it to you with zero context about whether the person talking knows what they're saying.

TubeScholar's whole reason to exist is the layer most tools skip: **should you trust this answer, and why?**

---

## ⭐ The Trust Layer

Every answer is meant to carry a trust assessment, not just text. The trust layer is built from three signals:

- **Claim verification**, are the specific claims supported by the retrieved context and by external reference sources?
- **Attribution and confidence**, how many independent sources back a claim, and is it a single creator's opinion or a broadly supported fact?
- **Channel reputation**, the speaker's relevant authority and the channel's track record.

These are surfaced **transparently** alongside the response, so the user sees the reasoning, not just a number.

### What's built today

- **A verification agent** that proof-reads each drafted answer and fact-checks claims against **Wikidata** and **Wikipedia**. For claims about a real person (occupation, field of work, education) it resolves the name to a Wikidata entity (QID) and compares structured facts; for events, places, and concepts it checks against the relevant Wikipedia article. It flags unsupported or contradicted claims with a `pass` / `revise` verdict rather than rewriting the answer.
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

## 🏗️ Architecture (target)

TubeScholar is a **multi-agent system** built on a LangGraph `StateGraph`. An intent classifier routes each turn before it reaches the supervisor, so cheap social turns skip retrieval entirely. The supervisor then coordinates sub-agents (retrieval, verification, relevance grading, trust scoring) instead of running everything through a single monolithic chain.

This diagram shows the finished shape of the system:

```
User query
   │
   ▼
┌──────────────┐
│ Classify     │  intent: new_request / follow_up / corpus_action / meta / chitchat
└─────┬────────┘
      │
      ├──▶ Chitchat node ──▶ direct reply (no retrieval)
      │
      └──▶ Supervisor
              │
              ├──▶ Retrieval tools ──▶ ChromaDB (vector search over transcripts)
              ├──▶ Corpus tools ──▶ ingest / search YouTube / list / delete videos
              ├──▶ Relevance grader ──▶ is the context good enough to answer from?
              ├──▶ Verification agent ──▶ Wikidata + Wikipedia fact-check
              └──▶ Trust scorer ──▶ reputation + attribution + claim confidence
                       │
                       ▼
                Grounded, trust-annotated answer  ──▶  Chainlit (token-streamed)
```

Transcripts are ingested via `youtube-transcript-api` (through rotating proxies), chunked, embedded with Gemini, and persisted in ChromaDB. When a video has no captions, the pipeline falls back to a metadata and top-comments overview (clearly labelled as inferred). An `openai-whisper` speech-to-text path exists but is gated off by default while audio transcription is offloaded to an external service.

---

## ✅ Current Progress

- **Ingestion pipeline**: YouTube Data API metadata, caption retrieval through Webshare rotating proxies, concurrent multi-URL ingestion on a bounded thread pool with per-thread HTTP clients (thread-safe), word-window chunking, Gemini embeddings, and persistent ChromaDB storage. Deterministic chunk IDs give idempotent re-ingestion, a 7-day staleness check skips up-to-date videos, and an in-memory registry is rebuilt from the store on startup.
- **Graceful failure handling**: every tool returns an agent-readable status dict (invalid URL, not found, quota/blocked, transcript unavailable, empty transcript) instead of crashing the run, including a metadata + top-comments fallback when no transcript exists.
- **Agent graph**: structured intent classification, a direct chitchat path, and a `langgraph_supervisor` supervisor wired with the full pipeline toolset plus the verification sub-agent. Conversation history is token-capped per thread and persisted via an async SQLite checkpointer.
- **Verification agent**: a tool-using agent backed by a Wikidata/Wikipedia client (`get_profile`, `search_person`, `get_property`, `humanise_qid`, `wiki_search`).
- **Streaming UI**: a Chainlit chat frontend with subgraph-aware token filtering, so only user-facing answer tokens are streamed (internal proof-reading stays hidden).
- **Evaluation harness**: a local Gemini-as-judge `RAGEvaluator` scoring precision, recall, faithfulness, and relevance, with versioned CSV output and LangSmith tracing.

---

## 🧰 Tech Stack

| Layer | Choice |
|-------|--------|
| **Orchestration** | LangGraph + LangChain (`langgraph_supervisor`) |
| **LLM** | `gemini-3.1-flash-lite` (all nodes: app, supervisor, agents, evaluator) |
| **Embeddings** | `models/gemini-embedding-001` |
| **Vector store** | ChromaDB (persistent) |
| **Transcripts** | `youtube-transcript-api` + `openai-whisper` (CPU fallback, currently gated off) |
| **Video metadata / search** | YouTube Data API v3 |
| **External fact-checking** | Wikidata + Wikipedia APIs |
| **Evaluation & tracing** | LangSmith tracing + custom Gemini-as-judge `RAGEvaluator` |
| **Frontend** | Chainlit (prototype); a TypeScript (Next.js) frontend over a FastAPI wrapper is planned |
| **Proxies** | Webshare (rotating, to avoid IP bans during ingestion) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com/), free tier is fine for development)
- A YouTube Data API v3 key
- (Optional) Webshare credentials for rotating proxies
- (Optional) A LangSmith API key for tracing and evaluation

### Setup

```bash
# clone
git clone https://github.com/<your-username>/tubescholar.git
cd tubescholar

# install
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
# the code reads PAID_GEMINI_API by default (swap to FREE_GEMINI_API in source if needed)
PAID_GEMINI_API=your_gemini_key
YOUTUBE_API_KEY=your_youtube_data_api_key

# optional
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
WEBSHARE_PROXY_USERNAME=...
WEBSHARE_PROXY_PASSWORD=...
WHISPER_ENABLED=0   # set to 1 to enable the local Whisper caption fallback
```

### Run

```bash
cd app/backend
chainlit run app.py -w
```

Then open the Chainlit link in your browser.

---

## 📊 Evaluation

Evaluation is treated as a **first-class signal**, not an afterthought.

- A custom **Gemini-as-judge** `RAGEvaluator` scores **precision, recall, faithfulness, and relevance** (each 0.0 to 1.0).
- Runs are versioned (`rag-v1`, `rag-v2`, ...) and written to local CSVs, with one averaged summary row per run. (RAGAS was dropped as incompatible with the modern LangChain stack.)
- The evaluator calls the vector store directly (bypassing the graph) to avoid LangChain format-string errors from curly braces in transcript chunks.
- LangSmith provides tracing across the pipeline via `@traceable`.

---

## 🗺️ Roadmap

**Stretch goals**
- [ ] Channel reputation scoring and a blended, explained trust score
- [ ] Dedicated relevance-grading agent and explicit usage-mode selector
- [ ] TypeScript (Next.js) frontend over a FastAPI wrapper
- [ ] Deployment via Hugging Face Spaces or a VPS
- [ ] Chrome extension

---

## 📄 License

Licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for the full text.
