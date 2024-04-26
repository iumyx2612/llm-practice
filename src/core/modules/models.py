from typing import Optional, List, Any, Sequence

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.base.llms.types import (
    LLMMetadata,
    ChatResponse,
    ChatMessage, CompletionResponse, ChatResponseGen, CompletionResponseGen, ChatResponseAsyncGen,
    CompletionResponseAsyncGen)
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback)
from llama_index.llms.openai import OpenAI

import google.generativeai as genai
from google.ai.generativelanguage import GenerateAnswerRequest


class GoogleEmbedding(BaseEmbedding):
    api_key: str = Field(description="API Key for Google Gemini")
    task_type: str = Field(default="retrieval_document")

    def __init__(self,
                 api_key: str,
                 model_name: str = "models/embedding-001",
                 task_type: str = "retrieval_document",
                 **kwargs):
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            task_type=task_type,
            **kwargs
        )
        genai.configure(api_key=self.api_key)

    def _get_query_embedding(self,
                             query: str,
                             title: Optional = None) -> Embedding:
        embedding = genai.embed_content(
            model=self.model_name,
            content=query,
            title=title
        )
        return embedding["embedding"]

    async def _aget_query_embedding(self,
                                    query: str,
                                    title: Optional = None) -> Embedding:
        return self._get_query_embedding(query, title)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._get_query_embedding(query=text)

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        embeddings = genai.embed_content(
            model=self.model_name,
            content=texts
        )
        return embeddings["embedding"]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        return self._get_text_embeddings(texts)

    def class_name(cls) -> str:
        return "GoogleEmbedding"


class GoogleLLM(LLM):
    api_key: str = Field(description="API Key for Google Gemini")
    model: str = Field(
        default="models/gemini-pro"
    )
    is_chat_model: bool = Field(
        default=False,
        description="Multi-turn conversation"
    )
    temperature: float = Field(
        default=0.0,
        description="Controls the randomness of the output",
        gte=0.0,
        lte=1.0,
    )
    max_output_tokens: Optional[int] = Field(
        description="Maximum number of output tokens available for this model",
        default=None
    )

    _llm = PrivateAttr()
    _chat_session = PrivateAttr()
    _model_data = PrivateAttr()

    def __init__(self,
                 api_key: str,
                 model: str = "models/gemini-pro",
                 temperature: float = 0.0,
                 max_output_tokens: int = None,
                 **kwargs) -> None:
        super().__init__(api_key=api_key,
                         model=model,
                         temperature=temperature,
                         max_output_tokens=max_output_tokens,
                         # base class
                         **kwargs)

        genai.configure(api_key=self.api_key)
        gen_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens)
        self._llm = genai.GenerativeModel(
            self.model, generation_config=gen_config)
        if self.is_chat_model:
            self._chat_session = self._llm.start_chat()
        self._model_data = genai.get_model(self.model)

    @classmethod
    def class_name(cls) -> str:
        return "google_gemini"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self._model_data.input_token_limit,
            num_output=self._model_data.output_token_limit,
            is_chat_model=self.is_chat_model,
            model_name=self._model_data.display_name,
        )

    def _chat(self, messages: str, **kwargs: Any) -> ChatResponse:
        response = self._chat_session.send_message(messages, **kwargs)
        gemini_message = ChatMessage(
            content=response.text
        )
        return ChatResponse(
            messages=gemini_message
        )

    @llm_chat_callback()
    def chat(self, messages: str, **kwargs: Any) -> ChatResponse:
        return self._chat(messages, **kwargs)

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise NotImplementedError("This class does not support streaming chat")

    def _complete(
            self, prompt: str
    ) -> CompletionResponse:
        response = self._llm.generate_content(prompt)
        return CompletionResponse(
            text=response.text
        )

    def _stream_complete(
            self, prompt: str
    ) -> ChatResponseGen:
        response = self._llm.generate_content(prompt, stream=True)

        def gen() -> ChatResponseGen:
            pass
        return gen()

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return self._complete(prompt)

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError("This class does not support streaming completion")

    # ==== Async Endpoints ====
    @llm_chat_callback()
    def achat(
        self, messages: str, **kwargs: Any
    ) -> ChatResponse:
        return self._chat(messages, **kwargs)

    @llm_chat_callback()
    def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError("This class does not support streaming chat")

    @llm_completion_callback()
    def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return self._complete(prompt)

    @llm_completion_callback()
    def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError("This class does not support streaming completion")

    # ==== General Generate Function ====
    def predict(
        self,
        prompt: str,
        **prompt_args: Any,
    ) -> str:
        if self.metadata.is_chat_model:
            output = self.chat(prompt)
        else:
            output = self.complete(prompt)
        response = output.text
        return response
