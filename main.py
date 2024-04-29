from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)
from dotenv import load_dotenv

from src.core.modules.models import GoogleLLM, GoogleEmbedding
from src.core.modules.response_synthesizers import google_response_synthesizer
from src.core.utils.settings import load_settings

load_dotenv("local.env")
app_settings = load_settings()
api_key = app_settings.google_ai.api_key
query = "Hello"


llm = GoogleLLM(api_key=api_key, temperature=0.0)
embed_model = GoogleEmbedding(api_key=api_key)

Settings.llm = llm
Settings.embed_model = embed_model

storage_context = StorageContext.from_defaults(
    persist_dir="index"
)
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(
    response_synthesizer=google_response_synthesizer(llm=llm)
)
response = query_engine.query(query)
print(response)