import pytest
from pydantic import BaseModel

from goose import flow
from goose._agent import Agent


class MyMessage(BaseModel):
    text: str


@flow
async def my_flow(*, message: MyMessage, agent: Agent) -> None:
    pass


@pytest.mark.asyncio
async def test_my_flow() -> None:
    async with my_flow.start_run(run_id="1") as run:
        await my_flow.generate(message=MyMessage(text="Hello"), agent=run.agent)

    async with my_flow.start_run(run_id="1") as run:
        await my_flow.generate(message=MyMessage(text="Hello"), agent=run.agent)
