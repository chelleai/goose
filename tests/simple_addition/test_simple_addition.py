from pathlib import Path

import pytest

from vers.flow import Flow
from vers.types.tasks import TaskOutput


class AdditionResult(TaskOutput):
    result: str


# @pytest.mark.asyncio
# async def test_simple_addition() -> None:
#     flow = Flow.load(
#         name="simple_addition",
#         response_models=[AdditionResult],
#         flows_spec_path=Path("tests/simple_addition/flow.yaml"),
#         tasks_spec_path=Path("tests/simple_addition/tasks.yaml"),
#     )
#     await flow.run(run_id="test_simple_addition", num1="1", num2="2")
#     task_output = flow.get("add_numbers", response_model=AdditionResult)

#     assert task_output.result == "3"

@pytest.mark.asyncio
async def test_simple_addition() -> None:
    flow = Flow.load(
        name="simple_addition",
        response_models=[AdditionResult],
        flows_spec_path=Path("tests/simple_addition/flow.yaml"),
        tasks_spec_path=Path("tests/simple_addition/tasks.yaml"),
    )
    try:
        await flow.run(run_id="test_simple_addition", num1="1", num2="2")
        task_output = flow.get("add_numbers", response_model=AdditionResult)
        assert task_output.result == "3"
    finally:
        await flow.cleanup()
