import typer
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings

from src.core.modules.models import GoogleEmbedding
from .settings import load_settings


def main(
        dotenv_path: str,
        document_folder: str = "data",
        persist_dir: str = "index",
        chunk_size: int = 250,
        chunk_overlap: int = 50
) -> None:
    load_dotenv(dotenv_path)
    settings = load_settings()
    emb_model = GoogleEmbedding(
        api_key=settings.google_ai.api_key
    )
    Settings.embed_model = emb_model
    documents = SimpleDirectoryReader(
        input_dir=document_folder
    ).load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    index.storage_context.persist(persist_dir=persist_dir)


if __name__ == '__main__':
    typer.run(main)