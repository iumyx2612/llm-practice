import typer
from typing import Optional
from typing_extensions import Annotated

from src.core.modules.models import GoogleLLM

DEFAULT_API_KEY = ""


def main(
        query: str,
        api_key: Annotated[Optional[str], typer.Argument()] = None,
        temp: float = 0.0
) -> None:
    if api_key is None:
        api_key = DEFAULT_API_KEY
    llm = GoogleLLM(api_key=api_key, temperature=temp)
    response = llm.predict(query)
    print(response)


if __name__ == '__main__':
    typer.run(main)
