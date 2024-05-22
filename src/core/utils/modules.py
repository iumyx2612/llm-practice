from llama_index.core import Settings

from src.core.modules.models import GoogleEmbedding, GoogleLLM
from .settings import Settings as AppSettings


def setup_modules(settings: AppSettings):
    api_key = settings.google_ai.api_key
    llm = GoogleLLM(api_key=api_key)
    embed_model = GoogleEmbedding(api_key=api_key)

    Settings.llm = llm
    Settings.embed_model = embed_model
