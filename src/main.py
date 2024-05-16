from typing import Union

import typer
from typing import Optional
from typing_extensions import Annotated
from dotenv import load_dotenv

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)

from src.core.modules.models import GoogleLLM, GoogleEmbedding
from src.core.modules.response_synthesizers import google_response_synthesizer
from src.core.utils.settings import load_settings

from fastapi import FastAPI

from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    query: str


@app.post('/predict')
def predict(data: Query,
            dotenv_path: str = "C:\\Users\ETC\Documents\maintn\llm-practice\example.env",
            #query: str,
            persist_dir: str = "index",
            temp: float = 0.0):
        load_dotenv(dotenv_path)
        settings = load_settings()
        api_key = settings.google_ai.api_key
        llm = GoogleLLM(api_key=api_key, temperature=temp)
        embed_model = GoogleEmbedding(api_key=api_key)
        Settings.llm = llm
        Settings.embed_model = embed_model

        storage_context = StorageContext.from_defaults(
            persist_dir=persist_dir
        )
        index = load_index_from_storage(storage_context)
        query_engine = index.as_query_engine(
            response_synthesizer=google_response_synthesizer(llm=llm)
        )
        response = query_engine.query(data)
        return response