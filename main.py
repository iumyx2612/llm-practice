from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)
from dotenv import load_dotenv

from src.core.modules.models import GoogleLLM, GoogleEmbedding
from src.core.modules.response_synthesizers import google_response_synthesizer
from src.core.utils.settings import load_settings
from fastapi import FastAPI,UploadFile, File

from pydantic import BaseModel
import shutil

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings

from src.core.modules.models import GoogleEmbedding
from src.core.utils.settings import load_settings
'''load_dotenv("C:\\Users\ETC\Documents\maintn\llm-practice\example.env")
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
print(response)'''

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
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
        response = query_engine.query(data.query)
        return response


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    upload_directory = "uploads/"
    os.makedirs(upload_directory, exist_ok=True)
    file_location = os.path.join(upload_directory, file.filename)

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    embedding(document_folder=upload_directory)

    return {"filename": file.filename}




def embedding(
        dotenv_path: str='C:\\Users\ETC\Documents\maintn\llm-practice\example.env',
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


@app.get("/", response_class=HTMLResponse)
async def read_index():
    return FileResponse("index.html")

