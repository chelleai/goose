import random
import string

import pytest

from goose import Result, flow, task


class GeneratedWord(Result):
    word: str


class Words(Result):
    words: list[GeneratedWord]


@task
async def generate_random_word(*, n_characters: int) -> GeneratedWord:
    return GeneratedWord(
        word="".join(random.sample(string.ascii_lowercase, n_characters))
    )


@task
async def duplicate_word(*, word: str, times: int) -> GeneratedWord:
    return GeneratedWord(word="".join([word] * times))


@flow
async def downstream_task() -> None:
    word = await generate_random_word(n_characters=10)
    await duplicate_word(word=word.word, times=10)


@pytest.mark.asyncio
async def test_downstream_task() -> None:
    with downstream_task.run() as state:
        await downstream_task.generate()
        print("flow state", state.dump())

    duplicated_word = state.get(task=duplicate_word)
    assert len(duplicated_word.result.word) == 100
