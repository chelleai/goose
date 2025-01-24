import random
import string

import pytest

from goose.core import Flow, task


@task
async def generate_random_word(*, n_characters: int) -> str:
    return "".join(random.sample(string.ascii_lowercase, n_characters))


@pytest.mark.asyncio
async def test_single_task() -> None:
    with Flow(name="random_word") as flow:
        word = generate_random_word(n_characters=10)

    await flow.generate()

    assert len(word.out) == 10
