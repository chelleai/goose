from pathlib import Path

import pytest

from vers.flow import Flow
from vers.types.tasks import TaskOutput

from typing import List


ARRAY_LOWER_LIMIT = 1
ARRAY_UPPER_LIMIT = 10
ADDEND = 2


class GenerateArrayResult(TaskOutput):
    result: List[str]

class ArrayAdditionResult(TaskOutput):
    result: str


@pytest.mark.asyncio
async def test_mapped_task_input_flow() -> None:
    flow = Flow.load(
        name="mapped_task_input_flow",
        response_models=[GenerateArrayResult, ArrayAdditionResult],
        flows_spec_path=Path("tests/mapped_task_input/flows.yaml"),
        tasks_spec_path=Path("tests/mapped_task_input/tasks.yaml"),
    )
    await flow.run(run_id="test_mapped_task_input_flow", num1=str(ARRAY_LOWER_LIMIT), num2=str(ARRAY_UPPER_LIMIT), num3=str(ADDEND))
    first_task_output = flow.get("generate_array_task", response_model=GenerateArrayResult)
    second_task_output = [flow.get("array_addition_task", response_model=ArrayAdditionResult, key=key).result for key in first_task_output.result]

    assert first_task_output.result == [str(k1) for k1 in range(ARRAY_LOWER_LIMIT, ARRAY_UPPER_LIMIT + 1)]
    assert second_task_output == [str(k1) for k1 in range(ARRAY_LOWER_LIMIT + ADDEND, ARRAY_UPPER_LIMIT + ADDEND + 1)]

