import random
import string
from unittest.mock import Mock

import pytest
from pytest_mock import MockerFixture

from goose import Result, flow, task
from goose.agent import SystemMessage, TextMessagePart, UserMessage


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
async def sentence() -> None:
    words = [await generate_random_word(n_characters=10) for _ in range(3)]
    await make_sentence(words=words)


@pytest.mark.asyncio
@pytest.mark.usefixtures("generate_random_word_adapter")
async def test_jamming() -> None:
    async with sentence.start_run(run_id="1") as first_run:
        await sentence.generate()

    initial_random_words = first_run.get_all(task=generate_random_word)
    assert len(initial_random_words) == 3

    # imagine this is a new process
    async with sentence.start_run(run_id="1") as second_run:
        await generate_random_word.refine(
            index=1,
            user_message=UserMessage(parts=[TextMessagePart(text="Change it")]),
            context=SystemMessage(parts=[TextMessagePart(text="Extra info")]),
        )

    random_words = second_run.get_all(task=generate_random_word)
    assert len(random_words) == 3
    assert random_words[0].result.word != "__ADAPTED__"  # not adapted
    assert random_words[1].result.word == "__ADAPTED__"  # adapted
    assert random_words[2].result.word != "__ADAPTED__"  # not adapted

    # imagine this is a new process
    async with sentence.start_run(run_id="1") as third_run:
        await sentence.generate()

    resulting_sentence = third_run.get(task=make_sentence)
    assert (
        resulting_sentence.result.sentence
        == f"{initial_random_words[0].result.word} __ADAPTED__ {initial_random_words[2].result.word}"
    )
