from typing import (
    Optional,
    Sequence,
    Callable,
)

from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.agent.react.output_parser import ReActOutputParser
from llama_index.core.callbacks import (
    CallbackManager,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.memory.types import BaseMemory
from llama_index.core.objects.base import ObjectRetriever
from llama_index.core.tools import BaseTool, ToolOutput
from llama_index.core.agent import ReActAgent

from .agent_worker import GeminiReActWorker

class GeminiReAct(ReActAgent):
    def __init__(
            self,
            tools: Sequence[BaseTool],
            llm: LLM,
            memory: BaseMemory,
            max_iterations: int = 10,
            react_chat_formatter: Optional[ReActChatFormatter] = None,
            output_parser: Optional[ReActOutputParser] = None,
            callback_manager: Optional[CallbackManager] = None,
            verbose: bool = False,
            tool_retriever: Optional[ObjectRetriever[BaseTool]] = None,
            context: Optional[str] = None,
            handle_reasoning_failure_fn: Optional[
                Callable[[CallbackManager, Exception], ToolOutput]
            ] = None,
    ):
        super().__init__(
            tools,
            llm,
            memory,
            max_iterations,
            react_chat_formatter,
            output_parser,
            callback_manager,
            verbose,
            tool_retriever,
            context,
            handle_reasoning_failure_fn
        )

        self.agent_worker = GeminiReActWorker.from_tools(
            tools=tools,
            tool_retriever=tool_retriever,
            llm=llm,
            max_iterations=max_iterations,
            react_chat_formatter=react_chat_formatter,
            output_parser=output_parser,
            callback_manager=callback_manager,
            verbose=verbose,
            handle_reasoning_failure_fn=handle_reasoning_failure_fn,
        )

        if callback_manager is not None:
            self.agent_worker.set_callback_manager(callback_manager)