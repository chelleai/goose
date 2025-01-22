from pathlib import Path

import pytest

from vers.flow import Flow
from vers.types.tasks import TaskOutput


WORD_TO_DEFINE = "congruent"


class DefineWordResult(TaskOutput):
    result: str


@pytest.mark.asyncio
async def test_define_word() -> None:
    flow = Flow.load(
        name="define_word_flow",
        response_models=[DefineWordResult],
        flows_spec_path=Path("tests/define_word/flows.yaml"),
        tasks_spec_path=Path("tests/define_word/tasks.yaml"),
    )
    await flow.run(run_id="test_define_word", word_to_define=WORD_TO_DEFINE)
    task_output = flow.get("define_word_task", response_model=DefineWordResult)

    assert WORD_TO_DEFINE.lower() in task_output.result.lower()
