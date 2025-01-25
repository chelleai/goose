import random
import string

import pytest

from goose.conversation import Conversation
from goose.core import Flow, Node, task
from goose.types import TextMessagePart, UserMessage


@task
async def generate_random_word(*, n_characters: int) -> str:
    return "".join(random.sample(string.ascii_lowercase, n_characters))


@generate_random_word.regenerator
async def regenerate_random_word(
    *, result: str, conversation: Conversation[str]
) -> str:
    return "Random word"


@task
async def duplicate_word(*, word: Node[str], times: int) -> str:
    return "".join([word.result] * times)


@duplicate_word.regenerator
async def regenerate_duplicate_word(
    *, result: str, conversation: Conversation[str]
) -> str:
    return "Regenerated " + result


@pytest.mark.asyncio
async def test_regenerate_no_downstream_nodes() -> None:
    with Flow(name="regenerate") as flow:
        word = generate_random_word(n_characters=10)
        duplicated_word = duplicate_word(word=word, times=10)

    await flow.generate()
    initial_result = duplicated_word.result

    await flow.regenerate(
        target=duplicated_word,
        message=UserMessage(parts=[TextMessagePart(text="regenerate this")]),
    )

    assert duplicated_word.result == "Regenerated " + initial_result


@pytest.mark.asyncio
async def test_regenerate_with_downstream_node() -> None:
    with Flow(name="regenerate") as flow:
        word = generate_random_word(n_characters=10)
        duplicated_word = duplicate_word(word=word, times=2)

    await flow.generate()

    await flow.regenerate(
        target=word,
        message=UserMessage(parts=[TextMessagePart(text="regenerate this")]),
    )

    assert word.result == "Random word"
    assert duplicated_word.result == "Random wordRandom word"
