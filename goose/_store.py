from __future__ import annotations

from typing import Protocol

from goose._flow import FlowRun
from goose._state import FlowRunState


class IFlowRunStore(Protocol):
    def __init__(self, *, flow_name: str) -> None: ...
    async def get(self, *, run_id: str) -> FlowRun | None: ...
    async def save(self, *, run: FlowRun) -> None: ...
    async def delete(self, *, run_id: str) -> None: ...


class InMemoryFlowRunStore(IFlowRunStore):
    def __init__(self, *, flow_name: str) -> None:
        self._flow_name = flow_name
        self._runs: dict[str, FlowRunState] = {}

    async def get(self, *, run_id: str) -> FlowRun | None:
        state = self._runs.get(run_id)
        if state is not None:
            return FlowRun.load(state)

    async def save(self, *, run: FlowRun) -> None:
        self._runs[run.id] = run.dump()

    async def delete(self, *, run_id: str) -> None:
        self._runs.pop(run_id, None)
