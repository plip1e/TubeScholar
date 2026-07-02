# Frontend (TypeScript)

Placeholder — the TypeScript frontend hasn't been scaffolded yet.

Planned: a Vite + TypeScript single-page app that talks to the FastAPI backend.

Next step (when we get here):
```bash
npm create vite@latest . -- --template vanilla-ts   # or react-ts
npm install
npm run dev            # serves on http://localhost:5173
```
Vite's dev server will proxy `/api/*` to the backend on `http://localhost:8000`.
See ../STRUCTURE.md for the full plan.