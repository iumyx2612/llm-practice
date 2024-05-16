import wikipedia
from dotenv import load_dotenv

from llama_index.core.tools import FunctionTool

from src.core.utils.settings import load_settings
from src.core.modules.models import GoogleLLM
from src.core.modules.agents import GeminiReAct


def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


def search_info_from_wikipedia(prompt: str) -> list[str]:
    """
    Do a wikipedia search for provided `prompt`
    Returns a list of ten wikipedia titles that matches the prompt
    """
    return wikipedia.search(prompt)


def get_wikipedia_content_from_title(title: str) -> str:
    """ Functions returns content of wikipedia page based on provided title """
    page = wikipedia.page(title)
    return page.content


if __name__ == '__main__':
    load_dotenv("../local.env")
    settings = load_settings()

    search_tool = FunctionTool.from_defaults(fn=search_info_from_wikipedia)
    content_tool = FunctionTool.from_defaults(fn=get_wikipedia_content_from_title)
    llm = GoogleLLM(
        api_key=settings.google_ai.api_key,
        is_chat_model=True
    )
    agent = GeminiReAct.from_tools(
        [search_tool, content_tool],
        llm=llm,
        verbose=True
    )

    response = agent.chat("Who is the Japanese singer Yuuri?")
    print(response.response)
    # multiply_tool = FunctionTool.from_defaults(fn=multiply)
    # agent = GeminiReAct.from_tools(
    #     [multiply_tool],
    #     llm=llm,
    #     verbose=True
    # )
    #
    # response = agent.chat("What is 2*4?")
    # print(response.response)