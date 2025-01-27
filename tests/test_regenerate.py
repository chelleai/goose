import random
import string

import pytest
from pydantic import BaseModel

from goose.conversation import Conversation
from goose.core import Flow, Node, task
from goose.types.agent import TextMessagePart, UserMessage


class GeneratedWord(BaseModel):
    word: str


@task
async def generate_random_word(*, n_characters: int) -> GeneratedWord:
    return GeneratedWord(
        word="".join(random.sample(string.ascii_lowercase, n_characters))
    )


@generate_random_word.regenerator
async def regenerate_random_word(
    *, result: GeneratedWord, conversation: Conversation[GeneratedWord]
) -> GeneratedWord:
    return GeneratedWord(word="Random word")


@task
async def duplicate_word(*, word: Node[GeneratedWord], times: int) -> GeneratedWord:
    return GeneratedWord(word="".join([word.result.word] * times))


@duplicate_word.regenerator
async def regenerate_duplicate_word(
    *, result: GeneratedWord, conversation: Conversation[GeneratedWord]
) -> GeneratedWord:
    return GeneratedWord(word="Regenerated " + result.word)


@pytest.mark.asyncio
async def test_regenerate_no_downstream_nodes() -> None:
    with Flow(name="regenerate", run_id="run1") as flow:
        word = generate_random_word(n_characters=10)
        duplicated_word = duplicate_word(word=word, times=10)

    await flow.generate()
    initial_result = duplicated_word.result

    await flow.regenerate(
        target=duplicated_word,
        message=UserMessage(parts=[TextMessagePart(text="regenerate this")]),
    )

    assert duplicated_word.result.word == "Regenerated " + initial_result.word


@pytest.mark.asyncio
async def test_regenerate_with_downstream_node() -> None:
    with Flow(name="regenerate", run_id="run1") as flow:
        word = generate_random_word(n_characters=10)
        duplicated_word = duplicate_word(word=word, times=2)

    await flow.generate()

    await flow.regenerate(
        target=word,
        message=UserMessage(parts=[TextMessagePart(text="regenerate this")]),
    )

    assert word.result.word == "Random word"
    assert duplicated_word.result.word == "Random wordRandom word"


@pytest.mark.asyncio
async def test_regenerate_adds_to_conversation() -> None:
    with Flow(name="regenerate", run_id="run1") as flow:
        word = generate_random_word(n_characters=10)
        duplicated_word = duplicate_word(word=word, times=2)

    await flow.generate()
    await flow.regenerate(
        target=duplicated_word,
        message=UserMessage(parts=[TextMessagePart(text="regenerate this")]),
    )
    assert duplicated_word.conversation is not None
    assert duplicated_word.conversation.user_messages == [
        UserMessage(parts=[TextMessagePart(text="regenerate this")])
    ]
    assert len(duplicated_word.conversation.results) == 2
