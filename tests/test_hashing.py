import random

import pytest

from goose import Agent, FlowArguments, Result, flow, task


class MyTaskResult(Result):
    rand: int


class MyFlowArguments(FlowArguments):
    n: int


@task
async def first_task(*, flow_arguments: MyFlowArguments) -> MyTaskResult:
    return MyTaskResult(rand=random.randint(0, 10000))


@task
async def second_task(*, flow_arguments: MyFlowArguments, rand: int) -> MyTaskResult:
    return MyTaskResult(rand=rand + flow_arguments.n)


@flow
async def my_flow(*, flow_arguments: MyFlowArguments, agent: Agent) -> None:
    first_result = await first_task(flow_arguments=flow_arguments)
    await second_task(flow_arguments=flow_arguments, rand=first_result.rand)


@pytest.mark.asyncio
async def test_my_flow() -> None:
    async with my_flow.start_run(run_id="1") as first_run:
        await my_flow.generate(MyFlowArguments(n=1))

    first_run_result = first_run.get_result(task=first_task)

    async with my_flow.start_run(run_id="1") as second_run:
        await my_flow.regenerate()

    second_run_result = second_run.get_result(task=first_task)
    assert second_run_result.rand == first_run_result.rand
