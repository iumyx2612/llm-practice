from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)

from src.core.modules.models import GoogleLLM, GoogleEmbedding

api_key = ""
query = "Who is Yuuri?"


llm = GoogleLLM(api_key=api_key, temperature=0.0)
embed_model = GoogleEmbedding(api_key=api_key)

Settings.llm = llm
Settings.embed_model = embed_model

storage_context = StorageContext.from_defaults(
    persist_dir="index"
)
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()
response = query_engine.query(query)
print(response)