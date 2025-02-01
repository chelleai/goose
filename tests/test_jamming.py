import random
import string

import pytest

from goose.agent import Agent, SystemMessage, TextMessagePart, UserMessage
from goose.flow import Conversation, Result, flow, task


class GeneratedWord(Result):
    word: str


class GeneratedSentence(Result):
    sentence: str


@task
async def generate_random_word(*, n_characters: int) -> GeneratedWord:
    return GeneratedWord(
        word="".join(random.sample(string.ascii_lowercase, n_characters))
    )


@generate_random_word.adapter
async def change_word(
    *, conversation: Conversation[GeneratedWord], agent: Agent
) -> GeneratedWord:
    return GeneratedWord(word="__ADAPTED__")


@task
async def make_sentence(*, words: list[GeneratedWord]) -> GeneratedSentence:
    return GeneratedSentence(sentence=" ".join([word.word for word in words]))


@flow
async def sentence() -> None:
    words = [await generate_random_word(n_characters=10) for _ in range(3)]
    await make_sentence(words=words)


@pytest.mark.asyncio
async def test_jamming() -> None:
    async with sentence.start_run(run_id="1") as first_run:
        await sentence.generate()

    initial_random_words = first_run.get_all(task=generate_random_word)
    assert len(initial_random_words) == 3

    # imagine this is a new process
    async with sentence.start_run(run_id="1") as second_run:
        await generate_random_word.jam(
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
