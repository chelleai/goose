from pathlib import Path

import pytest

from vers.flow import Flow
from vers.types.tasks import TaskOutput


class SimpleAdditionResult(TaskOutput):
    result: str


@pytest.mark.asyncio
async def test_simple_addition() -> None:
    flow = Flow.load(
        name="simple_addition_flow",
        response_models=[SimpleAdditionResult],
        flows_spec_path=Path("tests/simple_addition/flow.yaml"),
        tasks_spec_path=Path("tests/simple_addition/tasks.yaml"),
    )
    await flow.run(run_id="test_simple_addition", num1="1", num2="2")
    task_output = flow.get("simple_addition_task", response_model=SimpleAdditionResult)

    assert task_output.result == "3"

