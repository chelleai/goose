import random
import string

import pytest
from pydantic import BaseModel

from goose.core import Flow, task


class GeneratedWord(BaseModel):
    word: str


@task
async def generate_random_word(*, n_characters: int) -> GeneratedWord:
    return GeneratedWord(
        word="".join(random.sample(string.ascii_lowercase, n_characters))
    )


@pytest.mark.asyncio
async def test_single_task() -> None:
    with Flow(name="random_word", run_id="run1") as flow:
        word = generate_random_word(n_characters=10)

    await flow.generate()

    assert len(word.result.word) == 10
