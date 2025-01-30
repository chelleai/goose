import pytest
from pydantic import BaseModel

from goose.agent import (
    Agent,
    AgentResponse,
    GeminiModel,
    IAgentLogger,
    MediaMessagePart,
    TextMessagePart,
    UserMediaContentType,
    UserMessage,
    
)
from goose.flow import Result, flow, task


class CourseSyllabus(BaseModel):
    course_syllabus: str

class CourseSyllabusResponse(Result):
    course_syllabus: str


def get_llm_course_prompt(topic: str) -> str:
    return f"""Please generate a course syllabus using the provided material for a course titled {topic}.
        The course should be divided into six (6) modules. Each module should have a title, a brief description, and a list of topics to be covered.
        The course should also have a brief introduction and conclusion. Provide the syllabus in text format, NOT in JSON format."""

def upload_pdf_to_bytes(file_path: str) -> bytes:
    with open(file_path, 'rb') as file:
        pdf_bytes = file.read()
    return pdf_bytes

pdf_bytes = upload_pdf_to_bytes("tests/Linear algebra - Wikipedia.pdf")


text_message_part = TextMessagePart(text=get_llm_course_prompt("Linear Algebra Fundamentals"))
media_message_part = MediaMessagePart(content_type=UserMediaContentType.PDF, content=pdf_bytes)


@task
async def use_agent(*, agent: Agent) -> CourseSyllabusResponse:
    agent_response = await agent(
        messages=[UserMessage(parts=[text_message_part, media_message_part])],
        model=GeminiModel.FLASH_8B,
        response_model=CourseSyllabus,
        task_name="create_syllabus",
    )
    return CourseSyllabusResponse(course_syllabus=agent_response.course_syllabus)


@flow
async def agent_flow(*, agent: Agent) -> None:
    await use_agent(agent=agent)


class CustomLogger(IAgentLogger):
    logged_responses: list[AgentResponse[CourseSyllabusResponse]] = []

    async def __call__(self, *, response: AgentResponse[CourseSyllabusResponse]) -> None:
        self.logged_responses.append(response)


@flow(agent_logger=CustomLogger())
async def agent_flow_with_custom_logger(*, agent: Agent) -> None:
    await use_agent(agent=agent)


@pytest.mark.asyncio
async def test_agent() -> None:
    async with agent_flow.start_run(run_id="1") as run:
        await agent_flow.generate(agent=run.agent)

    generated_course_syllabus = run.get(task=use_agent).result.course_syllabus
    print(str(generated_course_syllabus))

    assert False


@pytest.mark.asyncio
async def test_agent_custom_logger() -> None:
    async with agent_flow_with_custom_logger.start_run(run_id="1") as run:
        await agent_flow_with_custom_logger.generate(agent=run.agent)

    assert len(CustomLogger.logged_responses) == 1
