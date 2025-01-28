import random
import string

import pytest

from goose.errors import Honk
from goose.flow import FlowRun, Result, flow, task


class GeneratedWord(Result):
    word: str


class GeneratedSentence(Result):
    sentence: str


@task
async def generate_random_word(*, n_characters: int) -> GeneratedWord:
    return GeneratedWord(
        word="".join(random.sample(string.ascii_lowercase, n_characters))
    )


@task
async def make_sentence(*, words: list[GeneratedWord]) -> GeneratedSentence:
    return GeneratedSentence(sentence=" ".join([word.word for word in words]))


@flow
async def with_state() -> None:
    word = await generate_random_word(n_characters=10)
    await make_sentence(words=[word])


@pytest.mark.asyncio
async def test_state_causes_caching() -> None:
    with with_state.start_run(name="1") as run:
        await with_state.generate()

    random_word = run.get(task=generate_random_word).result.word

    with pytest.raises(Honk):
        with_state.current_run

    loaded_run = FlowRun.load(run.dump())
    with with_state.start_run(name="2", run=loaded_run) as new_run:
        await with_state.generate()

    new_random_word = new_run.get(task=generate_random_word).result.word

    assert random_word == new_random_word  # unchanged node is not re-generated
