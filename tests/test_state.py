import random
import string
from unittest.mock import Mock

import pytest
from pytest_mock import MockerFixture

from goose import Agent, FlowArguments, Result, flow, task
from goose._internal.types.agent import SystemMessage, TextMessagePart, UserMessage
from goose.errors import Honk


class MyFlowArguments(FlowArguments):
    n_characters: int


class GeneratedWord(Result):
    word: str


class GeneratedSentence(Result):
    sentence: str


@task
async def generate_random_word(*, n_characters: int) -> GeneratedWord:
    return GeneratedWord(word="".join(random.sample(string.ascii_lowercase, n_characters)))


@pytest.fixture
def generate_random_word_adapter(mocker: MockerFixture) -> Mock:
    return mocker.patch.object(
        generate_random_word,
        "_Task__adapt",
        return_value=GeneratedWord(word="__ADAPTED__"),
    )


@task
async def make_sentence(*, words: list[GeneratedWord]) -> GeneratedSentence:
    return GeneratedSentence(sentence=" ".join([word.word for word in words]))


@flow
async def with_state(*, flow_arguments: MyFlowArguments, agent: Agent) -> None:
    word = await generate_random_word(n_characters=flow_arguments.n_characters)
    await make_sentence(words=[word])


@pytest.mark.asyncio
async def test_state_causes_caching() -> None:
    async with with_state.start_run(run_id="1") as run:
        await with_state.generate(MyFlowArguments(n_characters=10))

    random_word = run.get(task=generate_random_word).result.word

    with pytest.raises(Honk):
        with_state.current_run

    async with with_state.start_run(run_id="1") as new_run:
        await with_state.generate(MyFlowArguments(n_characters=10))

    new_random_word = new_run.get(task=generate_random_word).result.word

    assert random_word == new_random_word  # unchanged node is not re-generated


@pytest.mark.asyncio
@pytest.mark.usefixtures("generate_random_word_adapter")
async def test_state_undo() -> None:
    async with with_state.start_run(run_id="2"):
        await with_state.generate(MyFlowArguments(n_characters=10))

    async with with_state.start_run(run_id="2"):
        await generate_random_word.refine(
            index=0,
            user_message=UserMessage(parts=[TextMessagePart(text="Change it")]),
            context=SystemMessage(parts=[TextMessagePart(text="Extra info")]),
        )

    async with with_state.start_run(run_id="2") as run:
        generate_random_word.undo()

    assert run.get(task=generate_random_word).result.word != "__ADAPTED__"


@pytest.mark.asyncio
async def test_state_edit() -> None:
    async with with_state.start_run(run_id="3"):
        await with_state.generate(MyFlowArguments(n_characters=10))

    async with with_state.start_run(run_id="3") as run:
        generate_random_word.edit(result=GeneratedWord(word="__EDITED__"), index=0)

    assert run.get(task=generate_random_word).result.word == "__EDITED__"
