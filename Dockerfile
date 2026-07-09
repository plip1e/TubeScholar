# TubeScholar single-container image: FastAPI serves both the API (under
# /api) and the built React frontend (at /). Written for Hugging Face Spaces
# (docker SDK, app_port 7860) but nothing here is HF-specific; the same image
# runs on any Docker host.

# ---------- Stage 1: build the frontend ----------
# A separate stage means Node and node_modules never enter the final image;
# only the ~150 kB of built static files get copied out of it.
FROM node:22-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---------- Stage 2: the Python runtime ----------
FROM python:3.13-slim

# HF Spaces runs the container as UID 1000, not root. Create that user and
# make sure everything the app writes to (data/) is owned by it, otherwise
# Chroma/SQLite die on the first write with a permission error.
RUN useradd -m -u 1000 user

WORKDIR /app

# Install the backend package (deps first would need the package layout anyway
# since deps live in pyproject.toml, so this is one layer).
COPY pyproject.toml LICENSE ./
COPY backend/src ./backend/src
RUN pip install --no-cache-dir .

# The built frontend, where settings.static_dir expects it (frontend/dist).
COPY --from=frontend /build/dist ./frontend/dist

# Writable data dir for Chroma + the checkpoint DB. NOTE: on free HF Spaces
# this directory is EPHEMERAL: wiped on every restart/rebuild. With the paid
# persistent-storage add-on (mounts at /data), set the Space variables
# CHROMA_DIR=/data/chroma_db and CHECKPOINT_DB=/data/checkpoints.sqlite
# instead; no code change needed.
RUN mkdir -p /app/data && chown -R user:user /app/data

# OPTIONAL seed corpus: bake a starter video library into the image so the
# demo never wakes up empty (visitors' additions are still lost on restart).
# To enable: (1) uncomment the COPY below, (2) remove `data` from
# .dockerignore, (3) `git add -f data/chroma_db` in the Space repo. Be aware
# this publishes the ingested transcripts inside a public repo/image.
# COPY --chown=user data/chroma_db ./data/chroma_db

USER user

# The console script reads PORT; HF routes traffic to the port declared as
# app_port in the README front matter (7860).
ENV PORT=7860
EXPOSE 7860

CMD ["tubescholar"]
