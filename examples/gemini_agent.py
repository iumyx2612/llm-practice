import wikipedia
from dotenv import load_dotenv

from llama_index.core.tools import FunctionTool
from llama_index.core.utils import print_text

from src.core.utils.settings import load_settings
from src.core.modules.models import GoogleLLM
from src.core.modules.agents import GeminiReAct


SAMPLE_QUESTIONS = [
    "Who is Japanese singer Yuuri?",
    "Yuuri birthday"
]


def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


def search_info_from_wikipedia(prompt: str) -> list[str]:
    """
    Do a wikipedia search for provided `prompt`
    Returns a list of 5 wikipedia titles that matches the prompt
    """
    return wikipedia.search(prompt, results=5)


def get_wikipedia_content_from_title(title: str) -> str:
    """
    Functions returns content of wikipedia page based on provided title
    """
    page = wikipedia.page(title)
    return page.content


def ask_for_more_information(question: str) -> str:
    """
    Function to acquire more information
    """
    print_text(f"Please answer the LLM here: ", color='cyan')
    inp = input()
    return inp


if __name__ == '__main__':
    load_dotenv("local.env")
    settings = load_settings()

    search_tool = FunctionTool.from_defaults(fn=search_info_from_wikipedia)
    content_tool = FunctionTool.from_defaults(fn=get_wikipedia_content_from_title)
    inp_tool = FunctionTool.from_defaults(fn=ask_for_more_information)
    llm = GoogleLLM(
        api_key=settings.google_ai.api_key,
        is_chat_model=True
    )
    agent = GeminiReAct.from_tools(
        [search_tool, content_tool, inp_tool],
        llm=llm,
        verbose=True
    )

    response = agent.chat("Yuuri birthday")
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