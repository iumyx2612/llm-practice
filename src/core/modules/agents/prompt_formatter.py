from typing import Sequence, List, Optional

from llama_index.core.agent.react.formatter import (
    BaseAgentChatFormatter,
    get_react_tool_descriptions
)
from llama_index.core.agent.react.types import (
    BaseReasoningStep,
    ObservationReasoningStep,
)
from llama_index.core.tools import BaseTool
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.agent.react.prompts import REACT_CHAT_SYSTEM_HEADER
from google.generativeai.types import ContentsType, ContentDict

from ..utils import convert_chat_messages_to_ContentsType


class GoogleChatFormatter(BaseAgentChatFormatter):
    """
    ReAct chat formatter but applies to Google Gemini
    """
    system_header: str = REACT_CHAT_SYSTEM_HEADER
    context: str = ""

    def format(
        self,
        tools: Sequence[BaseTool],
        chat_history: List[ChatMessage],
        current_reasoning: Optional[List[BaseReasoningStep]] = None,
    ) -> ContentsType:
        current_reasoning = current_reasoning or []

        format_args = {
            "tool_desc": "\n".join(get_react_tool_descriptions(tools)),
            "tool_names": ", ".join([tool.metadata.get_name() for tool in tools]),
        }
        if self.context:
            format_args["context"] = self.context

        fmt_sys_header = self.system_header.format(**format_args)
        fmt_sys_header = [
            {
                "role": "user",
                "parts": [fmt_sys_header]
            }
        ]
        # format reasoning history as alternating user and assistant messages
        # where the assistant messages are thoughts and actions and the user
        # messages are observations
        chat_history = convert_chat_messages_to_ContentsType(chat_history)
        reasoning_history = fmt_sys_header
        reasoning_history.extend(chat_history)
        for reasoning_step in current_reasoning:
            if isinstance(reasoning_step, ObservationReasoningStep):
                message = ContentDict(
                    role="user",
                    parts=[reasoning_step.get_content()]
                )
            else:
                message = ContentDict(
                    role="model",
                    parts=[reasoning_step.get_content()]
                )
            reasoning_history.append(message)

        return reasoning_history
