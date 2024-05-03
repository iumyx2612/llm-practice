from pathlib import Path
import typer

from src.core.modules.crawl import save_text_from_wiki


# Crawl contents from a wikipedia page using wiki page id
def main(
        wiki_pageids: list[int],
        save_folder: Path = 'data/'
):
    save_text_from_wiki(wiki_pageids, save_folder)


if __name__ == '__main__':
    typer.run(main)
