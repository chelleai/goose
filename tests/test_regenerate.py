import random
import string

import pytest

from goose import Agent, FlowArguments, Result, flow, task
from goose.errors import Honk


class MyFlowArguments(FlowArguments):
    n_characters: int
    times: int


class GeneratedWord(Result):
    word: str


@task
async def generate_random_word(*, n_characters: int) -> GeneratedWord:
    word = "".join(random.sample(string.ascii_lowercase, n_characters))
    return GeneratedWord(word=word)


@task
async def duplicate_word(*, word: str, times: int) -> GeneratedWord:
    word = "".join([word] * times)
    return GeneratedWord(word=word)


@flow
async def flow_with_arguments(*, flow_arguments: MyFlowArguments, agent: Agent) -> None:
    word = await generate_random_word(n_characters=flow_arguments.n_characters)
    await duplicate_word(word=word.word, times=flow_arguments.times)


@pytest.mark.asyncio
async def test_flow_arguments_in_run() -> None:
    async with flow_with_arguments.start_run(run_id="1") as run:
        await flow_with_arguments.generate(MyFlowArguments(n_characters=10, times=10))

    async with flow_with_arguments.start_run(run_id="1") as run:
        await flow_with_arguments.regenerate()

    duplicated_word = run.get_result(task=duplicate_word)
    assert len(duplicated_word.word) == 100


@pytest.mark.asyncio
async def test_regenerate_before_generate_fails() -> None:
    with pytest.raises(Honk):
        async with flow_with_arguments.start_run(run_id="2"):
            await flow_with_arguments.regenerate()
