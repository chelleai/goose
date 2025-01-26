import random
import string

import pytest
from pydantic import BaseModel

from goose.core import Flow, Node, task


class GeneratedWord(BaseModel):
    word: str


@task
async def generate_random_word(*, n_characters: int) -> GeneratedWord:
    return GeneratedWord(
        word="".join(random.sample(string.ascii_lowercase, n_characters))
    )


@task
async def concatenate(*, words: list[Node[GeneratedWord]]) -> GeneratedWord:
    return GeneratedWord(word="".join([word.result.word for word in words]))


@pytest.mark.asyncio
async def test_map_reduce() -> None:
    with Flow(name="map_reduce", run_id="run1") as flow:
        words = [generate_random_word(n_characters=10) for _ in range(10)]
        concatenated_words = concatenate(words=words)

    await flow.generate()

    for word in words:
        assert word.result.word in concatenated_words.result.word

    assert len(concatenated_words.result.word) == 100
