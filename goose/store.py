from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from goose.flow import FlowRun


class IFlowRunStore(Protocol):
    def __init__(self, *, flow_name: str) -> None: ...
    async def get(self, *, run_id: str) -> FlowRun | None: ...
    async def save(self, *, run: FlowRun) -> None: ...


class InMemoryFlowRunStore(IFlowRunStore):
    def __init__(self, *, flow_name: str) -> None:
        self._flow_name = flow_name
        self._runs: dict[str, FlowRun] = {}

    async def get(self, *, run_id: str) -> FlowRun | None:
        return self._runs.get(run_id)

    async def save(self, *, run: FlowRun) -> None:
        self._runs[run.id] = run
