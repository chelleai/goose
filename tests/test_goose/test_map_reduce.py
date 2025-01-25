import random
import string

import pytest

from goose.core import Flow, Node, task


@task
async def generate_random_word(*, n_characters: int) -> str:
    return "".join(random.sample(string.ascii_lowercase, n_characters))


@task
async def concatenate(*, words: list[Node[str]]) -> str:
    return "".join([word.result for word in words])


@pytest.mark.asyncio
async def test_map_reduce() -> None:
    with Flow(name="map_reduce") as flow:
        words = [generate_random_word(n_characters=10) for _ in range(10)]
        concatenated_words = concatenate(words=words)

    await flow.generate()

    for word in words:
        assert word.result in concatenated_words.result

    assert len(concatenated_words.result) == 100
