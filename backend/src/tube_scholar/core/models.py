"""Provider-agnostic chat-model construction with clear error messages.

Thin wrapper over ``init_chat_model`` so that a missing/invalid API key surfaces
as one actionable line ("set GOOGLE_API_KEY in your .env") instead of a raw
pydantic ValidationError deep in the provider SDK.
"""

from langchain.chat_models import init_chat_model

# The standard environment variable each provider's SDK reads its key from.
PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "google_vertexai": "GOOGLE_APPLICATION_CREDENTIALS",
}


def chat_model(model: str, **kwargs):
    """Like ``init_chat_model(model, **kwargs)`` but with a provider-aware error.

    ``model`` is a ``provider:name`` string (e.g. ``"google_genai:gemini-3.1-flash-lite"``).
    On failure, re-raise with a hint naming the env var the chosen provider expects.
    """
    try:
        return init_chat_model(model, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        provider = model.split(":", 1)[0] if ":" in model else None
        env = PROVIDER_KEY_ENV.get(provider)
        hint = (
            f" If this is an auth error, set {env} in your .env "
            f"(provider inferred from MAIN_MODEL='{model}')."
            if env else
            f" Check that MAIN_MODEL='{model}' uses a valid 'provider:model' prefix "
            "and that the provider's package is installed."
        )
        raise RuntimeError(f"Failed to initialise chat model '{model}':{hint}\n  cause: {e}") from e
