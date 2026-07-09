# Frontend (React + TypeScript)

The TubeScholar web UI: a chat window that streams the agent's answers
token-by-token, plus a sidebar listing ingested videos with a form to ingest
new ones.

Built with **Vite 5** + **React 18** + **TypeScript** (Vite is pinned to v5
because newer majors require Node 20+ and this machine runs Node 18).

## Run it (dev)

Backend first, from the project root:

```bash
tubescholar          # uvicorn on http://localhost:8000
```

Then the frontend, from this folder:

```bash
npm install          # first time only
npm run dev          # Vite dev server on http://localhost:5173
```

Open http://localhost:5173. Vite proxies `/api/*` to the backend on `:8000`
(see `vite.config.ts`), so there are no CORS issues in dev.

## Layout

```
src/
├── main.tsx             # React root
├── App.tsx              # all state lives here, flows down as props
├── types.ts             # TS mirrors of the backend's Pydantic shapes
├── api/client.ts        # fetch wrappers + the SSE stream reader
├── components/
│   ├── ChatWindow.tsx   # scrolling message list
│   ├── MessageBubble.tsx
│   ├── Composer.tsx     # message input
│   └── VideoPanel.tsx   # sidebar: video library + ingest form
└── style.css            # dark theme, red accent, tokens on :root
```

## How streaming works

The backend's `POST /chat` responds with **Server-Sent Events**. The browser's
`EventSource` only supports GET, so `api/client.ts` reads the response body
with `fetch()` + `ReadableStream` and parses the SSE format by hand, yielding
tokens from an async generator that `App.tsx` consumes with `for await`.

## Build for production

```bash
npm run build        # type-checks (tsc) then bundles to dist/
```

In production there is no Node server: FastAPI serves `dist/` itself at `/`
(see `static_dir` in the backend settings), so the frontend and API are
same-origin and the Vite proxy only exists in dev.
