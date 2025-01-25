import asyncio
import contextvars
import inspect
from collections import defaultdict
from types import TracebackType
from typing import Any, Awaitable, Callable, Protocol, Self, overload

from graphlib import TopologicalSorter
from pydantic import BaseModel

from goose.agent import Agent
from goose.conversation import Conversation, ConversationState
from goose.regenerator import default_regenerator
from goose.types import AgentResponse, UserMessage


class NodeState[R: BaseModel](BaseModel):
    name: str
    conversation: ConversationState[R]

    @property
    def result(self) -> R:
        return self.conversation.results[-1]


class FlowState(BaseModel):
    nodes: list[NodeState[BaseModel]]


class NoResult:
    pass


class IRegenerator[R: BaseModel](Protocol):
    async def __call__(self, *, result: R, conversation: Conversation[R]) -> R: ...


class Task[**P, R: BaseModel]:
    def __init__(
        self, generator: Callable[P, Awaitable[R]], /, *, retries: int = 0
    ) -> None:
        self.retries = retries
        self._generator = generator
        self._regenerator: IRegenerator[R] = default_regenerator
        self._signature = inspect.signature(generator)
        self.__validate_fn()

    @property
    def result_type(self) -> type[R]:
        return_type = self._generator.__annotations__.get("return")
        if return_type is None:
            raise TypeError("Task must have a return type annotation")

        return return_type

    @property
    def name(self) -> str:
        return self._generator.__name__

    def regenerator(self, regenerator: IRegenerator[R], /) -> Self:
        self._regenerator = regenerator
        return self

    async def generate(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return await self._generator(*args, **kwargs)

    async def regenerate(self, *, result: R, conversation: Conversation[R]) -> R:
        return await self._regenerator(result=result, conversation=conversation)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> "Node[R]":
        arguments = self._signature.bind(*args, **kwargs).arguments
        return Node(task=self, arguments=arguments, result_type=self.result_type)

    def __validate_fn(self) -> None:
        if any(
            param.kind == inspect.Parameter.POSITIONAL_ONLY
            for param in self._signature.parameters.values()
        ):
            raise ValueError("Positional-only parameters are not supported in Tasks")


class Node[R: BaseModel]:
    def __init__(
        self,
        *,
        task: Task[Any, R],
        arguments: dict[str, Any],
        result_type: type[R],
        conversation: Conversation[R] | None = None,
    ) -> None:
        self.task = task
        self.arguments = arguments
        self.result_type = result_type
        self.conversation = conversation or Conversation[R]()
        self.name = task.name

        self._result: R | NoResult = NoResult()
        current_flow = Flow.get_current()
        if current_flow is None:
            raise RuntimeError("Cannot create a node without an active flow")
        self.id = current_flow.add_node(node=self)

    @property
    def has_result(self) -> bool:
        return not isinstance(self._result, NoResult)

    @property
    def result(self) -> R:
        if isinstance(self._result, NoResult):
            raise RuntimeError("Cannot access result of a node before it has run")
        return self._result

    async def generate(self) -> None:
        self._result = await self.task.generate(**self.arguments)
        self.conversation.replace_last_result(result=self.result)

    async def regenerate(self, *, message: UserMessage) -> None:
        self.conversation.add_message(message=message)
        self._result = await self.task.regenerate(
            result=self.result, conversation=self.conversation
        )
        self.conversation.add_result(result=self.result)

    def dump_state(self) -> NodeState[R]:
        return NodeState(name=self.name, conversation=self.conversation.dump())

    def load_state(self, *, state: NodeState[R]) -> None:
        self._result = state.result
        self.conversation = Conversation[self.result_type].load(
            state=state.conversation
        )

    def get_inbound_nodes(self) -> list["Node[BaseModel]"]:
        def __find_nodes(
            obj: Any, visited: set[int] | None = None
        ) -> list["Node[BaseModel]"]:
            if visited is None:
                visited = set()

            if isinstance(obj, Node):
                return [obj]
            elif isinstance(obj, dict):
                return [
                    node
                    for value in obj.values()
                    for node in __find_nodes(value, visited)
                ]
            elif isinstance(obj, list):
                return [node for item in obj for node in __find_nodes(item, visited)]
            elif isinstance(obj, tuple):
                return [node for item in obj for node in __find_nodes(item, visited)]
            elif isinstance(obj, set):
                return [node for item in obj for node in __find_nodes(item, visited)]
            elif hasattr(obj, "__dict__"):
                return [
                    node
                    for value in obj.__dict__.values()
                    for node in __find_nodes(value, visited)
                ]
            return []

        return __find_nodes(self.arguments)

    def __hash__(self) -> int:
        return hash(self.id)


class Flow:
    _current: contextvars.ContextVar["Flow | None"] = contextvars.ContextVar(
        "current_flow", default=None
    )

    def __init__(
        self,
        *,
        name: str,
        agent_logger: Callable[[AgentResponse[Any]], None] | None = None,
    ) -> None:
        self.name = name
        self._nodes: list[Node[BaseModel]] = []
        self._agent = Agent(flow_name=self.name, logger=agent_logger)

    @property
    def agent(self) -> Agent:
        return self._agent

    def dump_state(self) -> FlowState:
        return FlowState(nodes=[node.dump_state() for node in self._nodes])

    def load_state(self, *, flow_state: FlowState) -> None:
        nodes_by_name = {node.name: node for node in self._nodes}
        for node_state in flow_state.nodes:
            matching_node = nodes_by_name.get(node_state.name)
            if matching_node is None:
                raise RuntimeError(
                    f"Node {node_state.name} from state not found in flow"
                )

            matching_node.load_state(state=node_state)

    async def generate(self) -> None:
        graph = {node: node.get_inbound_nodes() for node in self._nodes}
        sorter = TopologicalSorter(graph)
        sorter.prepare()

        async with asyncio.TaskGroup() as task_group:
            while sorter.is_active():
                ready_nodes = list(sorter.get_ready())
                if ready_nodes:
                    for node in ready_nodes:
                        task_group.create_task(node.generate())
                    sorter.done(*ready_nodes)
                else:
                    await asyncio.sleep(0)

    async def regenerate(self, *, target: Node[Any], message: UserMessage) -> None:
        if not target.has_result:
            raise RuntimeError("Cannot regenerate a node without a result")

        await target.regenerate(message=message)

        # regenerate all downstream nodes
        full_graph = {node: node.get_inbound_nodes() for node in self._nodes}
        reversed_graph: dict[Node[BaseModel], set[Node[BaseModel]]] = defaultdict(set)
        for node, inbound_nodes in full_graph.items():
            for inbound_node in inbound_nodes:
                reversed_graph[inbound_node].add(node)

        subgraph: dict[Node[BaseModel], set[Node[BaseModel]]] = defaultdict(set)
        queue: list[Node[BaseModel]] = [target]

        while len(queue) > 0:
            node = queue.pop(0)
            outbound_nodes = reversed_graph[node]
            for outbound_node in outbound_nodes:
                subgraph[outbound_node].add(node)
                if outbound_node not in subgraph:
                    queue.append(outbound_node)

        if len(subgraph) > 0:
            sorter = TopologicalSorter(subgraph)
            sorter.prepare()

            async with asyncio.TaskGroup() as task_group:
                while sorter.is_active():
                    ready_nodes = list(sorter.get_ready())
                    if len(ready_nodes) > 0:
                        for node in ready_nodes:
                            if node != target:
                                task_group.create_task(node.generate())
                        sorter.done(*ready_nodes)
                    else:
                        await asyncio.sleep(0)

    @classmethod
    def get_current(cls) -> "Flow | None":
        return cls._current.get()

    def add_node(self, *, node: Node[Any]) -> str:
        existing_names = [node.name for node in self._nodes]
        number = sum(1 for name in existing_names if name == node.name)
        self._nodes.append(node)
        node_id = f"{node.name}_{number}"
        return node_id

    def __enter__(self) -> Self:
        if self._current.get() is not None:
            raise RuntimeError(
                "Cannot enter a new flow while another flow is already active"
            )
        self._current.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._current.set(None)


@overload
def task[**P, R: BaseModel](fn: Callable[P, Awaitable[R]], /) -> Task[P, R]: ...
@overload
def task[**P, R: BaseModel](
    *, retries: int = 0
) -> Callable[[Callable[P, Awaitable[R]]], Task[P, R]]: ...
def task[**P, R: BaseModel](
    fn: Callable[P, Awaitable[R]] | None = None, /, *, retries: int = 0
) -> Task[P, R] | Callable[[Callable[P, Awaitable[R]]], Task[P, R]]:
    if fn is None:
        return lambda fn: Task(fn, retries=retries)
    return Task(fn, retries=retries)
