
import os

GOOGLE_API_KEY = "AIzaSyCXn7vCgVa7hmfGlRYr-Dn71zw_VsKp51g"  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
def add_embedding(
        dotenv_path: str='D:\ETC\Work\llm-practice\example.envv',
        document_folder: str = "data",
        persist_dir: str = "index",
        chunk_size: int = 250,
        chunk_overlap: int = 50
) -> None:

    llm = Gemini(api_key=GOOGLE_API_KEY, model_name="models/gemini-pro")
    gemini_embedding_model = GeminiEmbedding(api_key=GOOGLE_API_KEY, model_name="models/embedding-001")

    # Set Global settings
    Settings.llm = llm
    Settings.embed_model = gemini_embedding_model
    # Load new documents
    print("Loading new documents...")
    new_documents = SimpleDirectoryReader(document_folder).load_data()
    index = VectorStoreIndex.from_documents(
        new_documents,
        show_progress=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    index.storage_context.persist(persist_dir=persist_dir)


    # Parse documents into nodes
    print("Parsing new documents into nodes...")
    for doc in new_documents:
        index.insert(doc)
        # Persist the index after inserting the new document
        index.storage_context.persist(persist_dir)
add_embedding(document_folder=r"D:\ETC\Work\llm-practice\uploads")