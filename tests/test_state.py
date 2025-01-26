import pytest
from pydantic import BaseModel

from goose.conversation import Conversation, ConversationState
from goose.core import Flow, FlowState, NodeState, task
from goose.types import TextMessagePart, UserMessage


class GeneratedWord(BaseModel):
    word: str


@task
async def generate_word() -> GeneratedWord:
    return GeneratedWord(word="the")


@generate_word.regenerator
async def regenerate_word(
    *, result: GeneratedWord, conversation: Conversation[GeneratedWord]
) -> GeneratedWord:
    return GeneratedWord(word="longer")


@pytest.mark.asyncio
async def test_dump_state() -> None:
    with Flow(name="test", run_id="run1") as flow:
        generate_word()

    await flow.generate()

    flow_state = flow.dump_state()
    assert len(flow_state.nodes) == 1
    assert flow_state.nodes[0].conversation.results == [GeneratedWord(word="the")]


@pytest.mark.asyncio
async def test_load_from_state():
    flow_state = FlowState(
        nodes=[
            NodeState(
                name="generate_word",
                conversation=ConversationState(
                    user_messages=[
                        UserMessage(parts=[TextMessagePart(text="longer")]),
                    ],
                    results=[GeneratedWord(word="a"), GeneratedWord(word="it")],
                ),
            )
        ]
    )

    with Flow(name="test", run_id="run1") as flow:
        word = generate_word()

    flow.load_state(flow_state=flow_state)

    assert len(word.conversation.results) == 2
    assert word.result.word == "it"

    await flow.regenerate(
        target=word, message=UserMessage(parts=[TextMessagePart(text="longer")])
    )

    assert word.result.word == "longer"
    assert len(word.conversation.results) == 3
