from pathlib import Path

import pytest

from vers.flow import Flow
from vers.types.tasks import TaskOutput


NUM_01 = 1
NUM_02 = 2
NUM_03 = 3


class FirstAdditionResult(TaskOutput):
    result: str

class SecondAdditionResult(TaskOutput):
    result: str

class SimpleAdditionResult(TaskOutput):
    result: str


@pytest.mark.asyncio
async def test_conditional_addition_flow_split() -> None:
    flow = Flow.load(
        name="conditional_addition_flow_split",
        response_models=[FirstAdditionResult, SecondAdditionResult],
        flows_spec_path=Path("tests/conditional_addition/flows.yaml"),
        tasks_spec_path=Path("tests/conditional_addition/tasks.yaml"),
    )
    await flow.run(run_id="test_conditional_addition_flow_split", num1=str(NUM_01), num2=str(NUM_02), num3=str(NUM_03))
    first_task_output = flow.get("first_simple_addition_task", response_model=FirstAdditionResult)
    second_task_output = flow.get("second_simple_addition_task", response_model=SecondAdditionResult)

    assert first_task_output.result == str(NUM_01 + NUM_02)
    assert second_task_output.result == str(NUM_01 + NUM_02 + NUM_03)

