from unittest.mock import Mock

import pytest
from aikernel import LLMMessagePart, LLMUserMessage, Router
from pytest_mock import MockerFixture

from goose import Agent, AgentResponse, FlowArguments, TextResult, flow, task
from goose._internal.agent import IAgentLogger


class TestFlowArguments(FlowArguments):
    pass


@pytest.fixture
def mock_llm_unstructured(mocker: MockerFixture) -> Mock:
    return mocker.patch(
        "goose._internal.agent.llm_unstructured",
        return_value=Mock(text="Hello", usage=Mock(input_tokens=10, output_tokens=10)),
    )


@task
async def use_agent(*, agent: Agent) -> TextResult:
    router = Router(
        model_list=[
            {"model_name": "gemini-2.0-flash-lite", "litellm_params": {"model": "gemini/gemini-2.0-flash-lite"}}
        ]
    )
    return await agent(
        messages=[LLMUserMessage(parts=[LLMMessagePart(content="Hello")])],
        model="gemini-2.0-flash-lite",
        task_name="greet",
        mode="generate",
        response_model=TextResult,
        router=router,
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
@pytest.mark.usefixtures("mock_llm_unstructured")
async def test_agent() -> None:
    async with agent_flow.start_run(run_id="1") as run:
        await agent_flow.generate(TestFlowArguments())

    assert run.get_result(task=use_agent).text == "Hello"


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_llm_unstructured")
async def test_agent_custom_logger() -> None:
    async with agent_flow_with_custom_logger.start_run(run_id="1"):
        await agent_flow_with_custom_logger.generate(TestFlowArguments())

    assert len(CustomLogger.logged_responses) == 1
