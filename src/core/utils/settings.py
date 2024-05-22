from typing import Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class TracingConfig(BaseModel):
    public_key: str
    secret_key: str
    user_id: str
    host: str = "https://cloud.langfuse.com"
    flush_at: int = 2


class GoogleAIConfig(BaseModel):
    api_key: str
    emb_model_name: str = "models/embedding-001"
    llm_model_name: str = "models/gemini-pro"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter='__')
    google_ai: Optional[GoogleAIConfig]
    tracing: Optional[TracingConfig]


def load_settings() -> Settings:
    return Settings()