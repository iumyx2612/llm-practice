from typing import Dict, Any, Iterable

from google.generativeai.types import ContentType, StrictContentType
from llama_index.core.llms.llm import LLMCompleteComponent


class GeminiCompleteComponent(LLMCompleteComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if "prompt" not in input:
            raise ValueError("Prompt must be in input dict.")

        if not (isinstance(input["prompt"], list) or
             isinstance(input["prompt"], Dict)):
            raise ValueError(f"Input {input} is not suitable for Gemini.")

        return input


