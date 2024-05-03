import typer
from typing import Optional
from typing_extensions import Annotated
from dotenv import load_dotenv

from src.core.modules.models import GoogleLLM
from src.core.utils.settings import load_settings


# Perform normal chat/generate functionality
def main(
        dotenv_path: str,
        query: str,
        temp: float = 0.0
) -> None:
    load_dotenv(dotenv_path)
    settings = load_settings()
    api_key = settings.google_ai.api_key
    llm = GoogleLLM(api_key=api_key, temperature=temp)
    response = llm.predict(query)
    print(response)


if __name__ == '__main__':
    typer.run(main)
