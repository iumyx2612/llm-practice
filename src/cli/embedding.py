import typer

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings

from src.core.utils import initialize

# Generate embedding of documents for RAG system
def main(
        dotenv_path: str,
        document_folder: str = "data",
        persist_dir: str = "index",
        chunk_size: int = 250,
        chunk_overlap: int = 50
) -> None:
    initialize(dotenv_path)

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
