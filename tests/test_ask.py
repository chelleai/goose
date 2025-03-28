from unittest.mock import Mock

import pytest
from aikernel import LLMAssistantMessage, LLMMessagePart, LLMUserMessage, Router
from pytest_mock import MockerFixture

from goose import Agent, FlowArguments, flow, task
from goose._internal.result import TextResult
from goose.errors import Honk

ROUTER = Router(
    model_list=[{"model_name": "gemini-2.0-flash-lite", "litellm_params": {"model": "gemini/gemini-2.0-flash-lite"}}]
)


class MockLiteLLMResponse:
    def __init__(self, *, response: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.choices = [Mock(message=Mock(content=response))]
        self.usage = Mock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


@pytest.fixture
def mock_llm_unstructured(mocker: MockerFixture) -> Mock:
    return mocker.patch(
        "goose._internal.agent.llm_unstructured",
        return_value=Mock(text="Here's the explanation!", usage=Mock(input_tokens=10, output_tokens=10)),
    )


class MyFlowArguments(FlowArguments):
    pass


@task
async def basic_task(*, flow_arguments: MyFlowArguments) -> TextResult:
    return TextResult(text="Hello, world!")


@flow
async def my_flow(*, flow_arguments: MyFlowArguments, agent: Agent) -> None:
    await basic_task(flow_arguments=flow_arguments)


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_llm_unstructured")
async def test_ask_adds_to_conversation():
    """Test that ask mode adds messages to conversation but doesn't change result"""

    async with my_flow.start_run(run_id="1") as run:
        await my_flow.generate(MyFlowArguments())

        # Get the initial result
        node_state = run.get_state(task=basic_task)
        original_result = node_state.raw_result

        # Ask a follow-up question
        response = await basic_task.ask(
            user_message=LLMUserMessage(parts=[LLMMessagePart(content="Can you explain how you got that?")]),
            router=ROUTER,
        )

        # Verify the response exists and makes sense
        assert response == "Here's the explanation!"

        # Get updated conversation
        node_state = run.get_state(task=basic_task)
        conversation = node_state.conversation

        # Verify that asking didn't change the original result
        assert node_state.raw_result == original_result

        # Verify the conversation includes the new messages
        assert len(conversation.user_messages) == 1
        assert len(conversation.assistant_messages) == 2  # Original result + response

        # Verify the last messages are our question and the response
        assert conversation.user_messages[-1].parts[0].content == "Can you explain how you got that?"
        assert isinstance(conversation.assistant_messages[-1], LLMAssistantMessage)
        assert conversation.assistant_messages[-1].parts[0].content == "Here's the explanation!"


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_llm_unstructured")
async def test_ask_requires_completed_task():
    """Test that ask mode only works on tasks that haven't been run"""

    async with my_flow.start_run(run_id="2") as run:
        # Set up flow arguments but don't run the task
        run.set_flow_arguments(MyFlowArguments())

        # Try to ask before running the task
        with pytest.raises(Honk, match="Cannot ask about a task that has not been initially generated"):
            await basic_task.ask(
                user_message=LLMUserMessage(parts=[LLMMessagePart(content="Can you explain?")]),
                router=ROUTER,
            )


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_llm_unstructured")
async def test_ask_multiple_questions():
    """Test that we can ask multiple follow-up questions"""

    async with my_flow.start_run(run_id="3") as run:
        await my_flow.generate(MyFlowArguments())

        # Ask several questions
        responses: list[str] = []
        questions = ["Why is that the answer?", "Can you explain it differently?", "What if we added 1 more?"]

        for question in questions:
            response = await basic_task.ask(
                user_message=LLMUserMessage(parts=[LLMMessagePart(content=question)]),
                router=ROUTER,
            )
            responses.append(response)

        # Verify we got responses for all questions
        assert len(responses) == len(questions)
        assert all(response == "Here's the explanation!" for response in responses)

        # Get conversation to verify messages
        node_state = run.get_state(task=basic_task)
        conversation = node_state.conversation

        # Verify the conversation includes all Q&A pairs
        # Should have: initial result + (question + answer for each question)
        assert len(conversation.user_messages) == len(questions)
        assert len(conversation.assistant_messages) == len(questions) + 1  # +1 for initial result
