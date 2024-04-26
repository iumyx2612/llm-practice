from typing import Union
from pathlib import Path
import os

import wikipedia


def save_text_from_wiki(
        page_ids: Union[int, list[int]],
        save_folder: Path
) -> None:
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if isinstance(page_ids, int):
        page_ids = [page_ids]
    for page_id in page_ids:
        wiki_text = wikipedia.page(pageid=page_id).content
        with open(f"{save_folder}/{page_id}.txt", 'w', encoding='utf-8') as f:
            f.write(wiki_text)
