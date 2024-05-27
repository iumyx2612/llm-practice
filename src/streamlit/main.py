import streamlit as st

from llama_index.core import (
    StorageContext,
    load_index_from_storage
)

from src.core.modules.response_synthesizers import google_response_synthesizer
from src.core.utils import initialize


def main(env_path: str):
    initialize(env_path)

    with st.sidebar:
        top_k = st.slider(
            label="Top-k documents",
            max_value=4,
            min_value=1,
            step=1
        )


    storage_context = StorageContext.from_defaults(
        persist_dir="index/"
    )
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(
        response_synthesizer=google_response_synthesizer(),
        similarity_top_k=top_k
    )

    with st.chat_message("user"):
        query = st.chat_input("Ask something here...")
        st.write(query)
    with st.chat_message("ai"):
        if query:
            response = query_engine.query(query)
            st.write(response.response)


if __name__ == '__main__':
    main("local.env")