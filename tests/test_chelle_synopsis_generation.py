import pytest
from pydantic import BaseModel

from goose.agent import (
    Agent,
    AgentResponse,
    GeminiModel,
    IAgentLogger,
    TextMessagePart,
    UserMessage,
)
from goose.flow import Result, flow, task


class CourseSynopsis(BaseModel):
    course_synopsis: str

class CourseSynopsisResponse(Result):
    course_synopsis: str

def get_llm_course_prompt(topic: str) -> str:
    return f"Please give me a short paragraph detailing topics to be covered in course on {topic}."


@task
async def use_agent(*, agent: Agent) -> CourseSynopsisResponse:
    agent_response = await agent(
        messages=[UserMessage(parts=[TextMessagePart(text=get_llm_course_prompt("Linear algebra fundamentals"))])],
        model=GeminiModel.FLASH_8B,
        response_model=CourseSynopsis,
        task_name="create_learning_synopsis",
    )
    return CourseSynopsisResponse(course_synopsis=agent_response.course_synopsis)


@flow
async def agent_flow(*, agent: Agent) -> None:
    await use_agent(agent=agent)


class CustomLogger(IAgentLogger):
    logged_responses: list[AgentResponse[CourseSynopsisResponse]] = []

    async def __call__(self, *, response: AgentResponse[CourseSynopsisResponse]) -> None:
        self.logged_responses.append(response)


@flow(agent_logger=CustomLogger())
async def agent_flow_with_custom_logger(*, agent: Agent) -> None:
    await use_agent(agent=agent)


@pytest.mark.asyncio
async def test_agent() -> None:
    async with agent_flow.start_run(run_id="1") as run:
        await agent_flow.generate(agent=run.agent)

    generated_course_synopsis = run.get(task=use_agent).result.course_synopsis
    print(str(generated_course_synopsis))

    assert False


@pytest.mark.asyncio
async def test_agent_custom_logger() -> None:
    async with agent_flow_with_custom_logger.start_run(run_id="1") as run:
        await agent_flow_with_custom_logger.generate(agent=run.agent)

    assert len(CustomLogger.logged_responses) == 1
