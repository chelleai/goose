import random
import string

import pytest

from goose import Agent, FlowArguments, Result, flow, task


class MyFlowArguments(FlowArguments):
    n_characters: int
    n_duplicates: int


class GeneratedWord(Result):
    word: str


@task
async def generate_random_word(*, n_characters: int) -> GeneratedWord:
    return GeneratedWord(word="".join(random.sample(string.ascii_lowercase, n_characters)))


@task
async def duplicate_word(*, word: str, times: int) -> GeneratedWord:
    return GeneratedWord(word="".join([word] * times))


@flow
async def downstream_task(*, flow_arguments: MyFlowArguments, agent: Agent) -> None:
    word = await generate_random_word(n_characters=flow_arguments.n_characters)
    await duplicate_word(word=word.word, times=flow_arguments.n_duplicates)


@pytest.mark.asyncio
async def test_downstream_task() -> None:
    async with downstream_task.start_run(run_id="1") as run:
        await downstream_task.generate(MyFlowArguments(n_characters=10, n_duplicates=10))

    duplicated_word = run.get_result(task=duplicate_word)
    assert len(duplicated_word.word) == 100
