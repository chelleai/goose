from unittest.mock import Mock

import pytest
from pytest_mock import MockerFixture

from goose import Agent, FlowArguments, TextResult, flow, task
from goose._internal.types.agent import MessagePart
from goose.agent import AgentResponse, AIModel, IAgentLogger, UserMessage


class TestFlowArguments(FlowArguments):
    pass


class MockLiteLLMResponse:
    def __init__(self, *, response: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.choices = [Mock(message=Mock(content=response))]
        self.usage = Mock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


@pytest.fixture
def mock_litellm(mocker: MockerFixture) -> Mock:
    return mocker.patch(
        "goose._internal.agent.acompletion",
        return_value=MockLiteLLMResponse(response="Hello", prompt_tokens=10, completion_tokens=10),
    )


@task
async def use_agent(*, agent: Agent) -> TextResult:
    return await agent(
        messages=[UserMessage(parts=[MessagePart(content="Hello")])],
        model=AIModel.GEMINI_FLASH_8B,
        task_name="greet",
    )


@flow
async def agent_flow(*, flow_arguments: TestFlowArguments, agent: Agent) -> None:
    await use_agent(agent=agent)


class CustomLogger(IAgentLogger):
    logged_responses: list[AgentResponse[TextResult]] = []

    async def __call__(self, *, response: AgentResponse[TextResult]) -> None:
        self.logged_responses.append(response)


@flow(agent_logger=CustomLogger())
async def agent_flow_with_custom_logger(*, flow_arguments: TestFlowArguments, agent: Agent) -> None:
    await use_agent(agent=agent)


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_litellm")
async def test_agent() -> None:
    async with agent_flow.start_run(run_id="1") as run:
        await agent_flow.generate(TestFlowArguments())

    assert run.get(task=use_agent).result.text == "Hello"


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_litellm")
async def test_agent_custom_logger() -> None:
    async with agent_flow_with_custom_logger.start_run(run_id="1"):
        await agent_flow_with_custom_logger.generate(TestFlowArguments())

    assert len(CustomLogger.logged_responses) == 1
