import random
import string
from unittest.mock import Mock

import pytest
from aikernel import LLMMessagePart, LLMSystemMessage, LLMUserMessage, Router
from pytest_mock import MockerFixture

from goose import Agent, FlowArguments, Result, flow, task
from goose._internal.result import FindReplaceResponse, Replacement
from goose.errors import Honk

ROUTER = Router(
    model_list=[{"model_name": "gemini-2.0-flash-lite", "litellm_params": {"model": "gemini/gemini-2.0-flash-lite"}}]
)


class MyFlowArguments(FlowArguments):
    pass


class GeneratedWord(Result):
    word: str


class GeneratedSentence(Result):
    sentence: str


@task
async def generate_random_word(*, n_characters: int) -> GeneratedWord:
    return GeneratedWord(word="".join(random.sample(string.ascii_lowercase, n_characters)))


@task
async def make_sentence(*, words: list[GeneratedWord]) -> GeneratedSentence:
    return GeneratedSentence(sentence=" ".join([word.word for word in words]))


@flow
async def sentence(*, flow_arguments: MyFlowArguments, agent: Agent) -> None:
    words = [await generate_random_word(n_characters=10) for _ in range(3)]
    await make_sentence(words=words)


@pytest.fixture
def mock_llm_structured(mocker: MockerFixture) -> Mock:
    return mocker.patch(
        "goose._internal.agent.llm_structured",
        return_value=Mock(
            structured_response=FindReplaceResponse(
                replacements=[Replacement(find="a", replace="b")],
            ),
            usage=Mock(input_tokens=10, output_tokens=10),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_llm_structured")
async def test_refining() -> None:
    async with sentence.start_run(run_id="1"):
        await sentence.generate(MyFlowArguments())

    async with sentence.start_run(run_id="1") as first_run:
        await generate_random_word.refine(
            index=0,
            user_message=LLMUserMessage(parts=[LLMMessagePart(content="Change it")]),
            context=LLMSystemMessage(parts=[LLMMessagePart(content="Extra info")]),
            router=ROUTER,
        )

    initial_random_words = first_run.get_all_results(task=generate_random_word)
    assert len(initial_random_words) == 3

    # imagine this is a new process
    async with sentence.start_run(run_id="1") as second_run:
        result = await generate_random_word.refine(
            user_message=LLMUserMessage(parts=[LLMMessagePart(content="Change it")]),
            context=LLMSystemMessage(parts=[LLMMessagePart(content="Extra info")]),
            router=ROUTER,
        )
        # Since refine now directly returns the result from the agent call
        assert isinstance(result, GeneratedWord)

    random_words = second_run.get_all_results(task=generate_random_word)
    assert len(random_words) == 3
    assert isinstance(random_words[0], GeneratedWord)
    assert isinstance(random_words[1], GeneratedWord)
    assert isinstance(random_words[2], GeneratedWord)


@pytest.mark.asyncio
async def test_refining_before_generate_fails() -> None:
    with pytest.raises(Honk):
        async with sentence.start_run(run_id="2"):
            await generate_random_word.refine(
                user_message=LLMUserMessage(parts=[LLMMessagePart(content="Change it")]),
                context=LLMSystemMessage(parts=[LLMMessagePart(content="Extra info")]),
                router=ROUTER,
            )
