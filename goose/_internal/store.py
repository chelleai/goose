from __future__ import annotations

from typing import Protocol

from .state import SerializedFlowRun


class IFlowRunStore(Protocol):
    def __init__(self, *, flow_name: str) -> None: ...
    async def get(self, *, run_id: str) -> SerializedFlowRun | None: ...
    async def save(self, *, run_id: str, run: SerializedFlowRun) -> None: ...
    async def delete(self, *, run_id: str) -> None: ...


class InMemoryFlowRunStore(IFlowRunStore):
    def __init__(self, *, flow_name: str) -> None:
        self._flow_name = flow_name
        self._runs: dict[str, SerializedFlowRun] = {}

    async def get(self, *, run_id: str) -> SerializedFlowRun | None:
        return self._runs.get(run_id)

    async def save(self, *, run_id: str, run: SerializedFlowRun) -> None:
        self._runs[run_id] = run

    async def delete(self, *, run_id: str) -> None:
        self._runs.pop(run_id, None)
