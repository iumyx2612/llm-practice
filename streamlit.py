import streamlit as st
import time

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


#import shutil

class Query(BaseModel):
    query: str


def qna(data: str,
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

def embedding(
        dotenv_path: str='C:\\Users\ETC\Documents\maintn\llm-practice\example.env',
        document_folder: str = "data",
        persist_dir: str = "index",
        chunk_size: int = 250,
        chunk_overlap: int = 50
) -> None:
    load_dotenv(dotenv_path)
    settings = load_settings()
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


def display_messages_from_history():
    """
    Displays chat messages from the history on app rerun.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
                st.markdown(message["content"])



def main():
    st.set_page_config(page_title="Chatbot", page_icon="💬", initial_sidebar_state="collapsed")
    
    left_column, central_column, right_column = st.columns([2, 1, 2])

    with left_column:
        st.write(" ")

    with central_column:
        st.image(str(r"C:\\Users\\ETC\Documents\\maintn\\rag-chatbot\\images\\bot-small.png"), use_column_width="auto")
        st.markdown("""<h4 style='text-align: center; color: grey;'></h4>""", unsafe_allow_html=True)

    with right_column:
        st.write(" ")
    with st.chat_message("assistant"):
        st.write("How can I help you today?")

    st.sidebar.title("Options")
    clear_button = st.sidebar.button("🗑️ Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = []
    display_messages_from_history()

    if user_input := st.chat_input("Input your question!"):

            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                #st.write("How can I help you today?")
                message_placeholder = st.empty()
                if user_input:
                    full_response = qna(user_input)
                    message_placeholder.markdown(full_response )
            
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a file to upload", type=["txt", "pdf", "docx"])
    print(uploaded_file)
    if uploaded_file is not None:
        try:
            upload_directory = "uploads/"
            os.makedirs(upload_directory, exist_ok=True)
            file_location = os.path.join(upload_directory, uploaded_file.name)
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(uploaded_file, buffer)
            st.sidebar.success(f"File {uploaded_file.name} uploaded and index updated successfully!")
            embedding(document_folder=upload_directory)

        except Exception as e:
            st.sidebar.error(f"An error occurred while uploading the file: {e}")

if __name__ == "__main__":
    main()