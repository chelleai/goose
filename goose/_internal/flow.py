from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from types import CodeType
from typing import Protocol, overload

from ..errors import Honk
from .agent import Agent, IAgentLogger
from .conversation import Conversation
from .result import Result
from .state import FlowArguments, FlowRun, get_current_flow_run, set_current_flow_run
from .store import IFlowRunStore, InMemoryFlowRunStore


class IGenerator[FlowArgumentsT: FlowArguments](Protocol):
    __name__: str

    async def __call__(self, *, flow_arguments: FlowArgumentsT, agent: Agent) -> None: ...


class IAdapter[ResultT: Result](Protocol):
    __code__: CodeType

    async def __call__(self, *, conversation: Conversation[ResultT], agent: Agent) -> ResultT: ...


class Flow[FlowArgumentsT: FlowArguments]:
    def __init__(
        self,
        fn: IGenerator[FlowArgumentsT],
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
    def flow_arguments_model(self) -> type[FlowArgumentsT]:
        arguments_model = self._fn.__annotations__.get("flow_arguments")
        if arguments_model is None:
            raise Honk("Flow function has an invalid signature. Must accept `flow_arguments` and `agent` as arguments.")

        return arguments_model

    @property
    def name(self) -> str:
        return self._name or self._fn.__name__

    @property
    def current_run(self) -> FlowRun[FlowArgumentsT]:
        run = get_current_flow_run()
        if run is None:
            raise Honk("No current flow run")
        return run

    @asynccontextmanager
    async def start_run(self, *, run_id: str) -> AsyncIterator[FlowRun[FlowArgumentsT]]:
        existing_serialized_run = await self._store.get(run_id=run_id)
        if existing_serialized_run is not None:
            run = FlowRun.load(
                serialized_flow_run=existing_serialized_run, flow_arguments_model=self.flow_arguments_model
            )
        else:
            run = FlowRun(flow_arguments_model=self.flow_arguments_model)

        old_run = get_current_flow_run()
        set_current_flow_run(run)

        run.start(flow_name=self.name, run_id=run_id, agent_logger=self._agent_logger)
        yield run
        await self._store.save(run_id=run_id, run=run.dump())
        run.end()

        set_current_flow_run(old_run)

    async def generate(self, flow_arguments: FlowArgumentsT, /) -> None:
        flow_run = get_current_flow_run()
        if flow_run is None:
            raise Honk("No current flow run")

        flow_run.set_flow_arguments(flow_arguments)
        await self._fn(flow_arguments=flow_arguments, agent=flow_run.agent)

    async def regenerate(self) -> None:
        flow_run = get_current_flow_run()
        if flow_run is None:
            raise Honk("No current flow run")

        await self._fn(flow_arguments=flow_run.flow_arguments, agent=flow_run.agent)


@overload
def flow[FlowArgumentsT: FlowArguments](fn: IGenerator[FlowArgumentsT], /) -> Flow[FlowArgumentsT]: ...
@overload
def flow[FlowArgumentsT: FlowArguments](
    *,
    name: str | None = None,
    store: IFlowRunStore | None = None,
    agent_logger: IAgentLogger | None = None,
) -> Callable[[IGenerator[FlowArgumentsT]], Flow[FlowArgumentsT]]: ...
def flow[FlowArgumentsT: FlowArguments](
    fn: IGenerator[FlowArgumentsT] | None = None,
    /,
    *,
    name: str | None = None,
    store: IFlowRunStore | None = None,
    agent_logger: IAgentLogger | None = None,
) -> Flow[FlowArgumentsT] | Callable[[IGenerator[FlowArgumentsT]], Flow[FlowArgumentsT]]:
    if fn is None:

        def decorator(fn: IGenerator[FlowArgumentsT]) -> Flow[FlowArgumentsT]:
            return Flow(fn, name=name, store=store, agent_logger=agent_logger)

        return decorator

    return Flow(fn, name=name, store=store, agent_logger=agent_logger)
