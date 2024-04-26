from typing import Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class GoogleAIConfig(BaseModel):
    api_key: str
    emb_model_name: str = "models/embedding-001"
    llm_model_name: str = "models/gemini-pro"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter='__')
    google_ai: Optional[GoogleAIConfig]


def load_settings() -> Settings:
    return Settings()