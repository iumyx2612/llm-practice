from typing import Dict

from llama_index.core import QueryBundle
from llama_index.core.utils import print_text
from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform

from src.core.modules.utils import convert_chat_messages_to_ContentsType


class GeminiHyDE(HyDEQueryTransform):
    def __init__(self,
                 *args,
                 verbose: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._verbose = verbose

    def _run(self, query_bundle: QueryBundle, metadata: Dict) -> QueryBundle:
        query_str = query_bundle.query_str
        prompt = self._hyde_prompt.format_messages(self._llm, context_str=query_str)
        prompt = convert_chat_messages_to_ContentsType(prompt)
        hypothetical_doc = self._llm.predict(prompt)
        if self._verbose:
            print_text(f">>> Hypothetical Doc: {hypothetical_doc} \n", color="magenta")
        embedding_strs = [hypothetical_doc]
        if self._include_original:
            embedding_strs.extend(query_bundle.embedding_strs)
        return QueryBundle(
            query_str=query_str,
            custom_embedding_strs=embedding_strs,
        )