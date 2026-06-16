# 🎓 TubeScholar

> A trustworthy research assistant for YouTube, not just another summarizer.

TubeScholar turns YouTube videos into a **queryable, source-aware knowledge base**. Ask a question, get an answer grounded in the actual transcript, *plus* a transparent trust signal that tells you how much you should rely on the source it came from.

The name is intentional: **Tube**(YouTube) + **Scholar**(Google Scholar), a serious research and education tool, not a TL;DR machine.

---

## The Problem

YouTube is one of the largest knowledge bases on the planet, but it's also unverified. Plain summarizers will happily condense a confident, wrong video into a clean paragraph and hand it to you with zero context about whether the person talking knows what they're saying.

TubeScholar's whole reason to exist is the layer most tools skip: **should you trust this answer, and why?**

---

## ⭐ The Trust Layer

Every answer carries a trust assessment built from three combined signals:

- **Credential checking**, does the speaker/channel have relevant authority on the topic?
- **Channel reputation scoring**, track record, signals of reliability vs. red flags.
- **AI claim confidence**, how well-supported is the specific claim by the retrieved context?

These are blended into a nuanced trust model and surfaced **transparently** alongside the response, so the user sees the reasoning, not just a number.

---

## 🧭 Usage Modes

| Mode | What it does |
|------|--------------|
| **Single Video** | Deep Q&A against one video's transcript. |
| **Topic Search** | Search YouTube for a topic, pull relevant videos, answer across them. |
| **Personal Collection** | Build and query your own curated corpus of videos *(core scope)*. |

---

## 🏗️ Architecture

TubeScholar is a **multi-agent system** built on a LangGraph `StateGraph`. A supervisor node routes user intent to the right path, retrieval, trust scoring, or other handling, rather than running everything through a single monolithic chain.

```
User query
   │
   ▼
┌─────────────┐
│ Supervisor  │  ── routes by intent
└─────┬───────┘
      │
      ├──▶ Retrieval node ──▶ ChromaDB (vector search over transcripts)
      │
      ├──▶ Trust scorer ──▶ credentials + reputation + claim confidence
      │
      └──▶ Other-intent node ──▶ (MultiQueryRetriever-backed)
                                          │
                                          ▼
                                   Grounded, trust-annotated answer
```

Transcripts are ingested via `youtube-transcript-api`, with an `openai-whisper` CPU fallback for videos without captions, then chunked and embedded into ChromaDB.

---

## 🧰 Tech Stack

| Layer | Choice |
|-------|--------|
| **Orchestration** | LangGraph + LangChain |
| **LLM** | `gemini-2.5-flash-lite` (main app), `gemini-2.5-flash` (heavier nodes, supervisor, trust scorer) |
| **Embeddings** | `models/gemini-embedding-001` |
| **Vector store** | ChromaDB |
| **Transcripts** | `youtube-transcript-api` + `openai-whisper` (CPU fallback) |
| **Video metadata / search** | YouTube Data API v3 |
| **Evaluation & tracing** | LangSmith + custom Gemini-as-judge `RAGEvaluator` |
| **Frontend** | Gradio |
| **Proxies** | webshare.io (rotating, to dodge IP bans during ingestion) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com/), free tier is fine for development)
- A YouTube Data API v3 key
- (Optional) webshare.io credentials for rotating proxies
- (Optional) A LangSmith API key for tracing/evaluation

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
GOOGLE_API_KEY=your_gemini_key
YOUTUBE_API_KEY=your_youtube_data_api_key

# optional
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
WEBSHARE_PROXY_USERNAME=...
WEBSHARE_PROXY_PASSWORD=...
```

### Run

```bash
python app.py
```

Then open the Gradio link in your browser.

---

## 📊 Evaluation

Evaluation is treated as a **first-class signal**, not an afterthought.

- Runs are versioned (`rag-v1`, `rag-v2`, …) and tracked in **LangSmith**.
- A custom **Gemini-as-judge** `RAGEvaluator` scores faithfulness, relevance, and answer quality. *(RAGAS was dropped, incompatible with the modern LangChain stack.)*
- The evaluator calls the vector store directly (bypassing the graph) to avoid LangChain format-string errors from curly braces in transcript chunks.

---

## 🗺️ Roadmap

**Stretch goals**
- [ ] Next.js frontend over a FastAPI wrapper (portfolio polish)
- [ ] Deployment via Hugging Face Spaces or a VPS
- [ ] Chrome extension

---

## 📌 Notes

This is a final-year project built for the IronHack bootcamp curriculum. The pipeline is currently prototyped against a Dark Souls 1 challenge-run video corpus as a test domain, the architecture itself is domain-agnostic.

---

## 📄 License

MIT *(or your choice, update this section)*.
