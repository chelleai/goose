import pytest

from goose import Agent, FlowArguments, flow


class MyFlowArguments(FlowArguments):
    text: str


@flow
async def my_flow(*, flow_arguments: MyFlowArguments, agent: Agent) -> None:
    pass


@pytest.mark.asyncio
async def test_my_flow() -> None:
    async with my_flow.start_run(run_id="1"):
        await my_flow.generate(MyFlowArguments(text="Hello"))

    async with my_flow.start_run(run_id="1"):
        await my_flow.generate(MyFlowArguments(text="Hello"))
