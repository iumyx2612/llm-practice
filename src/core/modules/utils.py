from typing import List

from llama_index.core.prompts import ChatMessage
from google.generativeai.types import ContentsType


def convert_chat_messages_to_ContentsType(messages: List[ChatMessage]) -> ContentsType:
    converted_msgs = []
    system_msg = ""
    for message in messages:
        if message.role == "user":
            converted_msgs.append(
                {
                    "role": "user",
                    "parts": [message.content]
                }
            )
        elif message.role == "assistant" or message.role == "chatbot":
            converted_msgs.append(
                {
                    "role": "model",
                    "parts": [message.content]
                }
            )
        elif message.role == "system":
            system_msg = message.content

    converted_msgs[0]['parts'].insert(0, f"*{system_msg}*")

    return converted_msgs
