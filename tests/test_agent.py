from unittest.mock import Mock

import pytest
from pydantic import BaseModel
from pytest_mock import MockerFixture

from goose.agent import Agent, GeminiModel, TextMessagePart, UserMessage
from goose.flow import Result, flow, task


class GreetingResult(Result):
    greeting: str


class Greeting(BaseModel):
    greeting: str


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
            model=Greeting(greeting="Hello"),
            prompt_tokens=10,
            completion_tokens=10,
        ),
    )


@task
async def use_agent(*, agent: Agent) -> GreetingResult:
    greeting = await agent(
        messages=[UserMessage(parts=[TextMessagePart(text="Hello")])],
        model=GeminiModel.FLASH_8B,
        response_model=Greeting,
        task_name="greet",
    )
    return GreetingResult(greeting=greeting.greeting)


@flow
async def agent_flow(*, agent: Agent) -> None:
    await use_agent(agent=agent)


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_litellm")
async def test_agent() -> None:
    with agent_flow.start_run(name="1") as run:
        await agent_flow.generate(agent=run.agent)

    assert run.get(task=use_agent).result.greeting == "Hello"
