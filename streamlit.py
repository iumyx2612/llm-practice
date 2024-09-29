import streamlit as st
import streamlit as st

import time

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)

from pydantic import BaseModel
import shutil


from pydantic import BaseModel
from dotenv import load_dotenv
import os

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings

from llama_index.core import StorageContext
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.gemini import GeminiEmbedding

from llama_index.llms.gemini import Gemini

#import shutil

class Query(BaseModel):
    query: str


GOOGLE_API_KEY = ""  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

from llama_index.llms.gemini import Gemini
from llama_index.core import PromptTemplate

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
    dir = os.path.dirname(os.path.abspath(__file__))+"\\"+persist_dir
    print(dir)
    print(len(os.listdir(dir)))
    if (len(os.listdir(dir))==0):
        index = VectorStoreIndex.from_documents(
            new_documents,
            show_progress=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        index.storage_context.persist(persist_dir=persist_dir)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        # Parse documents into nodes
        print("Parsing new documents into nodes...")
        for doc in new_documents:
            index.insert(doc)
            # Persist the index after inserting the new document
            index.storage_context.persist(persist_dir)

def question_and_answer(query,
                        persist_dir: str = "index",):

    template = (
        """ You are an assistant for question-answering tasks.
    Use the following context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use five sentences maximum and keep the answer concise.\n
    Question: {query_str} \nContext: {context_str} \nAnswer:"""
    )
    llm = Gemini(api_key=GOOGLE_API_KEY, model_name="models/gemini-pro")
    gemini_embedding_model = GeminiEmbedding(api_key=GOOGLE_API_KEY, model_name="models/embedding-001")

    # Set Global settings
    Settings.llm = llm
    Settings.embed_model = gemini_embedding_model
    llm_prompt = PromptTemplate(template)
    storage_context = StorageContext.from_defaults(
        persist_dir=persist_dir
    )
    index = load_index_from_storage(storage_context)
    # Query data from the persisted index
    query_engine = index.as_query_engine(text_qa_template=llm_prompt
        )
    response = query_engine.query(query)
    return response
        


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
        st.image(str(r"C:\Users\Mai\Downloads\8715124b521e29ea88ec80a1e5915b27.jpg"), use_column_width="auto")
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
                    full_response = question_and_answer(user_input)
                    message_placeholder.markdown(full_response )
            
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a file to upload", type=["txt", "pdf", "docx"])
    print('============>',uploaded_file)
    if uploaded_file is not None:
            upload_directory = "uploads/"
            os.makedirs(upload_directory, exist_ok=True)
            file_location = os.path.join(upload_directory, uploaded_file.name)
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(uploaded_file, buffer)
            st.sidebar.success(f"File {uploaded_file.name} uploaded and index updated successfully!")
            add_embedding(document_folder=upload_directory)

if __name__ == "__main__":
    main()
