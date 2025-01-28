import random
import string

import pytest

from goose import ConversationState, FlowState, Result, flow, task
from goose.agent import TextMessagePart, UserMessage


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
    *, conversation_state: ConversationState[GeneratedWord]
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
async def test_adapting() -> None:
    with sentence.run() as first_state:
        await sentence.generate()

    initial_random_words = first_state.get_all(task=generate_random_word)
    assert len(initial_random_words) == 3

    # imagine this is a new process
    second_state = FlowState.load(first_state.dump())
    with sentence.run(state=second_state):
        await generate_random_word.adapt(
            flow_state=second_state,
            index=1,
            user_message=UserMessage(parts=[TextMessagePart(text="Change it")]),
        )

    random_words = second_state.get_all(task=generate_random_word)
    assert len(random_words) == 3
    assert random_words[0].result.word != "__ADAPTED__"  # not adapted
    assert random_words[1].result.word == "__ADAPTED__"  # adapted
    assert random_words[2].result.word != "__ADAPTED__"  # not adapted

    # imagine this is a new process
    third_state = FlowState.load(second_state.dump())
    with sentence.run(state=third_state):
        await sentence.generate()

    resulting_sentence = third_state.get(task=make_sentence)
    assert (
        resulting_sentence.result.sentence
        == f"{initial_random_words[0].result.word} __ADAPTED__ {initial_random_words[2].result.word}"
    )
