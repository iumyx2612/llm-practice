import typer
from typing import Optional
from typing_extensions import Annotated

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)

from src.core.modules.models import GoogleLLM, GoogleEmbedding
from src.core.modules.response_synthesizers import google_response_synthesizer

DEFAULT_API_KEY = ""


def main(
        query: str,
        api_key: Annotated[Optional[str], typer.Argument()] = None,
        persist_dir: str = "index",
        temp: float = 0.0
) -> None:
    if api_key is None:
        api_key = DEFAULT_API_KEY
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
    response = query_engine.query(query)
    print(response)


if __name__ == '__main__':
    typer.run(main)