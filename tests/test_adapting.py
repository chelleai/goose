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
    with sentence.run() as state:
        await sentence.generate()

    loaded_state = FlowState.load(state.dump())
    with sentence.run(state=loaded_state) as new_state:
        await generate_random_word.adapt(
            flow_state=new_state,
            index=1,
            user_message=UserMessage(parts=[TextMessagePart(text="Change it")]),
        )

    random_words = new_state.get_all(task=generate_random_word)
    assert len(random_words) == 3
    assert random_words[0].result.word != "__ADAPTED__"
    assert random_words[1].result.word == "__ADAPTED__"
    assert random_words[2].result.word != "__ADAPTED__"
