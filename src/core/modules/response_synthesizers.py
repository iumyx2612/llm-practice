from typing import Generator, Any, Optional, Callable
import logging

from llama_index.core.utils import truncate_text
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.types import RESPONSE_TEXT_TYPE
from llama_index.core.prompts.default_prompt_selectors import (
    DEFAULT_REFINE_PROMPT_SEL,
    DEFAULT_TEXT_QA_PROMPT_SEL,
    DEFAULT_TREE_SUMMARIZE_PROMPT_SEL,
)
from llama_index.core.prompts.default_prompts import DEFAULT_SIMPLE_INPUT_PROMPT
from llama_index.core.service_context_elements.llm_predictor import LLMPredictorType
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.service_context import ServiceContext
from llama_index.core.service_context_elements.llm_predictor import LLMPredictorType
from llama_index.core.settings import (
    Settings,
    callback_manager_from_settings_or_context,
    llm_from_settings_or_context,
)
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.types import BasePydanticProgram

import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)
logger = logging.getLogger(__name__)


class GoogleCompactAndRefine(CompactAndRefine):

    def _give_response_single(
        self,
        query_str: str,
        text_chunk: str,
        **response_kwargs: Any,
    ) -> str:
        """Give response over chunks."""
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        text_chunks = self._prompt_helper.repack(text_qa_template, [text_chunk])

        response: Optional[RESPONSE_TEXT_TYPE] = None
        for cur_text_chunk in text_chunks:
            if response is None and not self._streaming:
                text_qa_prompt = self._text_qa_template.format(
                    context_str=cur_text_chunk,
                    query_str=query_str
                )
                response = self._llm.predict(text_qa_prompt)
            elif response is None and self._streaming:
                response = self._llm.stream(
                    text_qa_template,
                    context_str=cur_text_chunk,
                    **response_kwargs,
                )
            else:
                response = self._refine_response_single(
                    response,
                    query_str,
                    cur_text_chunk,
                    **response_kwargs,
                )
        if response is None:
            response = "Empty Response"
        if isinstance(response, str):
            response = response or "Empty Response"
        return response

    def _refine_response_single(
        self,
        response: str,
        query_str: str,
        text_chunk: str,
        **response_kwargs: Any,
    ) -> Optional[str]:
        fmt_text_chunk = truncate_text(text_chunk, 50)
        logger.debug(f"> Refine context: {fmt_text_chunk}")
        if self._verbose:
            print(f"> Refine context: {fmt_text_chunk}")

        refine_template = self._refine_template.partial_format(
            query_str=query_str, existing_answer=response
        )
        avail_chunk_size = self._prompt_helper._get_available_chunk_size(
            refine_template
        )

        if avail_chunk_size < 0:
            # if the available chunk size is negative, then the refine template
            # is too big and we just return the original response
            return response

        # obtain text chunks to add to the refine template
        text_chunks = self._prompt_helper.repack(
            refine_template, text_chunks=[text_chunk]
        )

        for cur_text_chunk in text_chunks:
            if not self._streaming:
                refine_prompt = self._refine_template.format(
                    query_str=query_str,
                    context_msg=cur_text_chunk,
                    existing_answer=response
                )
                response = self._llm.predict(refine_prompt)
            else:
                # TODO: structured response not supported for streaming
                if isinstance(response, Generator):
                    response = "".join(response)

                refine_template = self._refine_template.partial_format(
                    query_str=query_str, existing_answer=response
                )

                response = self._llm.stream(
                    refine_template,
                    context_msg=cur_text_chunk,
                    **response_kwargs,
                )

        return response


def google_response_synthesizer(
        llm: Optional[LLMPredictorType] = None,
        prompt_helper: Optional[PromptHelper] = None,
        service_context: Optional[ServiceContext] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        refine_template: Optional[BasePromptTemplate] = None,
        summary_template: Optional[BasePromptTemplate] = None,
        simple_template: Optional[BasePromptTemplate] = None,
        callback_manager: Optional[CallbackManager] = None,
        use_async: bool = False,
        streaming: bool = False,
        structured_answer_filtering: bool = False,
        output_cls: Optional[BaseModel] = None,
        program_factory: Optional[Callable[[PromptTemplate], BasePydanticProgram]] = None,
        verbose: bool = False,
) -> GoogleCompactAndRefine:
    text_qa_template = text_qa_template or DEFAULT_TEXT_QA_PROMPT_SEL
    refine_template = refine_template or DEFAULT_REFINE_PROMPT_SEL
    simple_template = simple_template or DEFAULT_SIMPLE_INPUT_PROMPT
    summary_template = summary_template or DEFAULT_TREE_SUMMARIZE_PROMPT_SEL

    callback_manager = callback_manager or callback_manager_from_settings_or_context(
        Settings, service_context
    )
    llm = llm or llm_from_settings_or_context(Settings, service_context)

    if service_context is not None:
        prompt_helper = service_context.prompt_helper
    else:
        prompt_helper = (
            prompt_helper
            or Settings._prompt_helper
            or PromptHelper.from_llm_metadata(
                llm.metadata,
            )
        )

    return GoogleCompactAndRefine(
        llm=llm,
        callback_manager=callback_manager,
        prompt_helper=prompt_helper,
        text_qa_template=text_qa_template,
        refine_template=refine_template,
        output_cls=output_cls,
        streaming=streaming,
        structured_answer_filtering=structured_answer_filtering,
        program_factory=program_factory,
        verbose=verbose,
        # deprecated
        service_context=service_context,
    )
