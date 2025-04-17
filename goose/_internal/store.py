from __future__ import annotations

from typing import Protocol


class IFlowRunStore(Protocol):
    def __init__(self, *, flow_name: str) -> None: ...
    async def get(self, *, run_id: str) -> str | None: ...
    async def save(self, *, run_id: str, run: str) -> None: ...
    async def delete(self, *, run_id: str) -> None: ...


class InMemoryFlowRunStore(IFlowRunStore):
    def __init__(self, *, flow_name: str) -> None:
        self._flow_name = flow_name
        self._runs: dict[str, str] = {}

    async def get(self, *, run_id: str) -> str | None:
        return self._runs.get(run_id)

    async def save(self, *, run_id: str, run: str) -> None:
        self._runs[run_id] = run

    async def delete(self, *, run_id: str) -> None:
        self._runs.pop(run_id, None)
