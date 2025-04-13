"""Flow module for orchestrating sequences of tasks.

This module provides the Flow class and flow decorator for building
workflows that coordinate multiple tasks with state management,
persistence, and run tracking.
"""

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
    """Protocol for flow generator functions.

    This protocol defines the interface for functions that can be used
    to create flows.

    Type Parameters:
        FlowArgumentsT: Type of flow arguments
    """

    __name__: str

    async def __call__(self, *, flow_arguments: FlowArgumentsT, agent: Agent) -> None: ...


class IAdapter[ResultT: Result](Protocol):
    """Protocol for adapters that transform conversations to results.

    This protocol defines the interface for functions that can adapt
    conversations to structured results.

    Type Parameters:
        ResultT: Type of result
    """

    __code__: CodeType

    async def __call__(self, *, conversation: Conversation, agent: Agent) -> ResultT: ...


class Flow[FlowArgumentsT: FlowArguments]:
    """Orchestrates a sequence of tasks with persistent state.

    A Flow manages the execution of tasks, tracks state, and handles
    persistence of results. It provides mechanisms for starting runs,
    generating results, and regenerating flows.

    Type Parameters:
        FlowArgumentsT: Type of flow arguments
    """

    def __init__(
        self,
        fn: IGenerator[FlowArgumentsT],
        /,
        *,
        name: str | None = None,
        store: IFlowRunStore | None = None,
        agent_logger: IAgentLogger | None = None,
    ) -> None:
        """Initialize a Flow.

        Args:
            fn: The flow generator function
            name: Optional custom name for the flow (defaults to function name)
            store: Optional store for persisting flow runs
            agent_logger: Optional logger for agent responses
        """
        self._fn = fn
        self._name = name
        self._agent_logger = agent_logger
        self._store = store or InMemoryFlowRunStore(flow_name=self.name)

    @property
    def flow_arguments_model(self) -> type[FlowArgumentsT]:
        """Get the flow arguments model type.

        Returns:
            The FlowArguments type used by this flow

        Raises:
            Honk: If the flow function has an invalid signature
        """
        arguments_model = self._fn.__annotations__.get("flow_arguments")
        if arguments_model is None:
            raise Honk("Flow function has an invalid signature. Must accept `flow_arguments` and `agent` as arguments.")

        return arguments_model

    @property
    def name(self) -> str:
        """Get the name of the flow.

        Returns:
            The name of the flow (custom name or function name)
        """
        return self._name or self._fn.__name__

    @property
    def current_run(self) -> FlowRun[FlowArgumentsT]:
        """Get the current flow run.

        Returns:
            The current flow run

        Raises:
            Honk: If there is no current flow run
        """
        run = get_current_flow_run()
        if run is None:
            raise Honk("No current flow run")
        return run

    @asynccontextmanager
    async def start_run(self, *, run_id: str) -> AsyncIterator[FlowRun[FlowArgumentsT]]:
        """Start a new run of this flow.

        This context manager starts a new flow run or loads an existing one,
        sets it as the current run, and handles persistence when the context exits.

        Args:
            run_id: Unique identifier for this run

        Yields:
            The flow run instance

        Example:
            ```python
            async with my_flow.start_run(run_id="123") as run:
                await my_flow.generate(MyFlowArguments(...))
            ```
        """
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
        """Execute the flow with the given arguments.

        This method runs the flow function with the provided arguments,
        executing all tasks defined within the flow.

        Args:
            flow_arguments: Arguments for the flow

        Raises:
            Honk: If there is no current flow run
        """
        flow_run = get_current_flow_run()
        if flow_run is None:
            raise Honk("No current flow run")

        flow_run.set_flow_arguments(flow_arguments)
        await self._fn(flow_arguments=flow_arguments, agent=flow_run.agent)

    async def regenerate(self) -> None:
        """Regenerate the flow using the same arguments.

        This method re-executes the flow function with the same arguments
        as the previous execution, which is useful for retrying a flow.

        Raises:
            Honk: If there is no current flow run
        """
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
    """Decorator for creating Flow instances.

    This decorator transforms async functions into Flow objects that can
    orchestrate task execution, manage state, and handle persistence.

    Examples:
        ```python
        @flow
        async def my_workflow(*, flow_arguments: MyFlowArguments, agent: Agent) -> None:
            # Flow implementation...

        # Or with parameters
        @flow(name="custom_name", store=CustomStore())
        async def my_workflow(*, flow_arguments: MyFlowArguments, agent: Agent) -> None:
            # Flow implementation...
        ```

    Args:
        fn: The function to decorate
        name: Optional custom name for the flow
        store: Optional store for persisting flow runs
        agent_logger: Optional logger for agent responses

    Returns:
        A Flow instance, or a decorator function if used with parameters
    """
    if fn is None:

        def decorator(fn: IGenerator[FlowArgumentsT]) -> Flow[FlowArgumentsT]:
            return Flow(fn, name=name, store=store, agent_logger=agent_logger)

        return decorator

    return Flow(fn, name=name, store=store, agent_logger=agent_logger)
