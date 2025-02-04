from __future__ import annotations

from typing import Protocol

from goose._internal.flow import FlowRun
from goose._internal.state import FlowArguments, SerializedFlowRun


class IFlowRunStore[FlowArgumentsT: FlowArguments](Protocol):
    def __init__(self, *, flow_name: str, flow_arguments_model: type[FlowArgumentsT]) -> None: ...
    async def get(self, *, run_id: str) -> FlowRun[FlowArgumentsT] | None: ...
    async def save(self, *, run_id: str, run: SerializedFlowRun) -> None: ...
    async def delete(self, *, run_id: str) -> None: ...


class InMemoryFlowRunStore[FlowArgumentsT: FlowArguments](IFlowRunStore[FlowArgumentsT]):
    def __init__(self, *, flow_name: str, flow_arguments_model: type[FlowArgumentsT]) -> None:
        self._flow_name = flow_name
        self._flow_arguments_model = flow_arguments_model
        self._runs: dict[str, SerializedFlowRun] = {}

    async def get(self, *, run_id: str) -> FlowRun[FlowArgumentsT] | None:
        serialized_flow_run = self._runs.get(run_id)
        if serialized_flow_run is not None:
            return FlowRun.load(
                serialized_flow_run=serialized_flow_run, flow_arguments_model=self._flow_arguments_model
            )

    async def save(self, *, run_id: str, run: SerializedFlowRun) -> None:
        self._runs[run_id] = run

    async def delete(self, *, run_id: str) -> None:
        self._runs.pop(run_id, None)
