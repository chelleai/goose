from unittest.mock import Mock

import pytest
from pydantic import BaseModel
from pytest_mock import MockerFixture

from goose.agent import Agent
from goose.core import Flow, Node, task
from goose.types import AgentResponse, GeminiModel, TextMessagePart, UserMessage


class MockLiteLLMResponse:
    def __init__(
        self, *, model: BaseModel, prompt_tokens: int, completion_tokens: int
    ) -> None:
        self.choices = [Mock(message=Mock(content=model.model_dump_json()))]
        self.usage = Mock(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )


@pytest.fixture
def mock_litellm(mocker: MockerFixture) -> Mock:
    return mocker.patch(
        "goose.agent.acompletion",
        return_value=MockLiteLLMResponse(
            model=GeneratedPerson(name="John", age=30),
            prompt_tokens=10,
            completion_tokens=10,
        ),
    )


class GeneratedPerson(BaseModel):
    name: str
    age: int


@task
async def generate_person(*, agent: Agent) -> GeneratedPerson:
    return await agent(
        messages=[UserMessage(parts=[TextMessagePart(text="Generate a person")])],
        model=GeminiModel.PRO,
        response_model=GeneratedPerson,
        task_name="generate_person",
    )


@task
async def double_age(*, person: Node[GeneratedPerson]) -> int:
    return person.result.age * 2


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_litellm")
async def test_use_agent_uses_response() -> None:
    with Flow(name="test_use_agent") as flow:
        person = generate_person(agent=flow.agent)
        doubled_age = double_age(person=person)

    await flow.generate()

    assert doubled_age.result == 60


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_litellm")
async def test_use_agent_logs_response() -> None:
    logs: list[str] = []

    def logger(response: AgentResponse[GeneratedPerson]) -> None:
        logs.append(response.model_dump_json())

    with Flow(name="test_use_agent", agent_logger=logger) as flow:
        generate_person(agent=flow.agent)

    await flow.generate()

    assert len(logs) == 1
    assert "John" in logs[0]
    assert "30" in logs[0]
