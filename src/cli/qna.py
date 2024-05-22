import typer

from llama_index.core import (
    StorageContext,
    load_index_from_storage
)

from src.core.modules.response_synthesizers import google_response_synthesizer

from src.core.utils import initialize

# Perform question and answer using RAG system
def main(
        dotenv_path: str,
        query: str,
        persist_dir: str = "index"
) -> None:
    initialize(dotenv_path)

    storage_context = StorageContext.from_defaults(
        persist_dir=persist_dir
    )
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(
        response_synthesizer=google_response_synthesizer()
    )
    response = query_engine.query(query)
    print(response.response)


if __name__ == '__main__':
    typer.run(main)
