from contextlib import asynccontextmanager
from types import CodeType
from typing import AsyncIterator, Awaitable, Callable, Protocol, overload

from goose._agent import Agent, IAgentLogger
from goose._conversation import Conversation
from goose._result import Result
from goose._state import FlowRun, get_current_flow_run, set_current_flow_run
from goose._store import IFlowRunStore, InMemoryFlowRunStore
from goose.errors import Honk


class IAdapter[ResultT: Result](Protocol):
    __code__: CodeType

    async def __call__(
        self, *, conversation: Conversation[ResultT], agent: Agent
    ) -> ResultT: ...


class Flow[**P]:
    def __init__(
        self,
        fn: Callable[P, Awaitable[None]],
        /,
        *,
        name: str | None = None,
        store: IFlowRunStore | None = None,
        agent_logger: IAgentLogger | None = None,
    ) -> None:
        self._fn = fn
        self._name = name
        self._agent_logger = agent_logger
        self._store = store or InMemoryFlowRunStore(flow_name=self.name)

    @property
    def name(self) -> str:
        return self._name or self._fn.__name__

    @property
    def current_run(self) -> FlowRun:
        run = get_current_flow_run()
        if run is None:
            raise Honk("No current flow run")
        return run

    @asynccontextmanager
    async def start_run(self, *, run_id: str) -> AsyncIterator[FlowRun]:
        existing_run = await self._store.get(run_id=run_id)
        if existing_run is None:
            run = FlowRun()
        else:
            run = existing_run

        old_run = get_current_flow_run()
        set_current_flow_run(run)

        run.start(flow_name=self.name, run_id=run_id, agent_logger=self._agent_logger)
        yield run
        await self._store.save(run=run)
        run.end()

        set_current_flow_run(old_run)

    async def generate(self, *args: P.args, **kwargs: P.kwargs) -> None:
        flow_run = get_current_flow_run()
        if flow_run is None:
            raise Honk("No current flow run")

        flow_run.set_flow_inputs(*args, **kwargs)
        await self._fn(*args, **kwargs)

    async def regenerate(self) -> None:
        flow_run = get_current_flow_run()
        if flow_run is None:
            raise Honk("No current flow run")

        flow_args, flow_kwargs = flow_run.flow_inputs
        await self._fn(*flow_args, **flow_kwargs)


@overload
def flow[**P](fn: Callable[P, Awaitable[None]], /) -> Flow[P]: ...
@overload
def flow[**P](
    *,
    name: str | None = None,
    store: IFlowRunStore | None = None,
    agent_logger: IAgentLogger | None = None,
) -> Callable[[Callable[P, Awaitable[None]]], Flow[P]]: ...
def flow[**P](
    fn: Callable[P, Awaitable[None]] | None = None,
    /,
    *,
    name: str | None = None,
    store: IFlowRunStore | None = None,
    agent_logger: IAgentLogger | None = None,
) -> Flow[P] | Callable[[Callable[P, Awaitable[None]]], Flow[P]]:
    if fn is None:

        def decorator(fn: Callable[P, Awaitable[None]]) -> Flow[P]:
            return Flow(fn, name=name, store=store, agent_logger=agent_logger)

        return decorator

    return Flow(fn, name=name, store=store, agent_logger=agent_logger)
