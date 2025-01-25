import random
import string

import pytest

from goose.core import Flow, Node, task


@task
async def generate_random_word(*, n_characters: int) -> str:
    return "".join(random.sample(string.ascii_lowercase, n_characters))


@task
async def duplicate_word(*, word: Node[str], times: int) -> str:
    return "".join([word.result] * times)


@pytest.mark.asyncio
async def test_downstream_task() -> None:
    with Flow(name="downstream_task") as flow:
        word = generate_random_word(n_characters=10)
        duplicated_word = duplicate_word(word=word, times=10)

    await flow.generate()

    assert len(duplicated_word.result) == 100
