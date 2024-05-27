import typer

from llama_index.core import (
    StorageContext,
    load_index_from_storage
)
from llama_index.core.query_engine import TransformQueryEngine

from src.core.modules.response_synthesizers import google_response_synthesizer
from src.core.modules.query_components.hyde import GeminiHyDE

from src.core.utils import initialize

# Perform question and answer using RAG system
def main(
        dotenv_path: str,
        query: str,
        persist_dir: str = "index",
        top_k: int = 4,
        use_hyde: bool = False
) -> None:
    initialize(dotenv_path)

    storage_context = StorageContext.from_defaults(
        persist_dir=persist_dir
    )
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(
        response_synthesizer=google_response_synthesizer(),
        similarity_top_k=top_k
    )
    if use_hyde:
        hyde = GeminiHyDE(include_original=True)
        query_engine = TransformQueryEngine(query_engine,
                                            query_transform=hyde)
    response = query_engine.query(query)
    print(response.response)


if __name__ == '__main__':
    typer.run(main)
