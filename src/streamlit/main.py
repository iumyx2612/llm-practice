import streamlit as st

from dotenv import load_dotenv

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)

from src.core.modules.models import GoogleLLM, GoogleEmbedding
from src.core.modules.response_synthesizers import google_response_synthesizer
from src.core.utils.settings import (
    load_settings,
    Settings as AppSettings
)


def main(env_path: str):
    load_dotenv(env_path)
    app_settings = load_settings()

    with st.sidebar:
        temperature = st.slider(
            label="Temperature",
            max_value=1.0,
            min_value=0.0,
            step=0.1
        )

    embed_model = GoogleEmbedding(
        api_key=app_settings.google_ai.api_key,
        model_name=app_settings.google_ai.emb_model_name
    )
    llm = GoogleLLM(
        api_key=app_settings.google_ai.api_key,
        model=app_settings.google_ai.llm_model_name,
        temperature=temperature
    )

    Settings.embed_model = embed_model
    Settings.llm = llm

    storage_context = StorageContext.from_defaults(
        persist_dir="index/"
    )
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(
        response_synthesizer=google_response_synthesizer(llm=llm)
    )

    with st.chat_message("user"):
        query = st.chat_input("Ask something here...")
        st.write(query)
    with st.chat_message("ai"):
        if query:
            response = query_engine.query(query)
            st.write(response)


if __name__ == '__main__':
    main("local.env")