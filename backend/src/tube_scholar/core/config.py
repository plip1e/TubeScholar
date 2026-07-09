"""Central configuration for the whole backend.

One place that reads environment variables (from a real ``.env`` file in dev, or
real env vars in production) and exposes them as a typed object. Import
``settings`` anywhere instead of scattering ``os.getenv(...)`` across the code.

Provider API keys are intentionally NOT modelled here. TubeScholar is
provider-agnostic: the ``MAIN_MODEL`` / ``EMBEDDING_MODEL`` strings carry a
``provider:model`` prefix, and each provider's SDK reads its own standard key from
the environment (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``GOOGLE_API_KEY``).
We call ``load_dotenv()`` below so those keys (living in ``.env`` during dev)
land in ``os.environ`` where the SDKs look for them. (pydantic-settings alone
reads ``.env`` into this object, not into ``os.environ``.)
"""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into os.environ so provider SDKs (OpenAI/Anthropic/Google) find their keys.
load_dotenv()


class Settings(BaseSettings):
    # --- models (provider:model strings, understood by init_chat_model / init_embeddings) ---
    # e.g. "openai:gpt-4o", "anthropic:claude-sonnet-4-5", "google_genai:gemini-3.1-flash-lite"
    main_model: str = "google_genai:gemini-3.1-flash-lite"        # MAIN_MODEL
    # e.g. "openai:text-embedding-3-small", "google_genai:models/gemini-embedding-001"
    embedding_model: str = "google_genai:models/gemini-embedding-001"  # EMBEDDING_MODEL

    # --- non-LLM API keys (these ARE passed explicitly, so we model them) ---
    youtube_api_key: str | None = None   # YOUTUBE_API_KEY (YouTube Data API v3)

    # --- LangSmith tracing (all optional) ---
    langsmith_tracing: bool = False      # LANGSMITH_TRACING
    langsmith_endpoint: str | None = None
    langsmith_api_key: str | None = None
    langsmith_project: str | None = None

    # --- Webshare proxy (optional, for transcript fetching) ---
    webshare_proxy_username: str | None = None
    webshare_proxy_password: str | None = None

    # --- storage paths (relative to the project root / cwd the app runs from) ---
    chroma_dir: str = "data/chroma_db"              # CHROMA_DIR
    checkpoint_db: str = "data/checkpoints.sqlite"  # CHECKPOINT_DB

    # --- built frontend (served by FastAPI when the directory exists) ---
    static_dir: str = "frontend/dist"               # STATIC_DIR

    # --- feature flags ---
    whisper_enabled: int = 0             # WHISPER_ENABLED (1 = local audio transcription)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore env vars we haven't modelled (proxy URLs, provider keys, ...)
    )


# Import this singleton elsewhere: `from tube_scholar.core.config import settings`
settings = Settings()
