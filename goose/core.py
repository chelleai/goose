import asyncio
import contextvars
import inspect
from types import TracebackType
from typing import Any, Awaitable, Callable, Protocol, Self, overload

from graphlib import TopologicalSorter

from goose.types.regenerator import Conversation


class NoResult:
    pass


class IRegenerator[R](Protocol):
    async def __call__(self, *, result: R, conversation: Conversation[R]) -> R: ...


class Node[R]:
    def __init__(
        self,
        *,
        task: "Task[Any, R]",
        arguments: dict[str, Any],
        conversation: Conversation[R] | None = None,
    ) -> None:
        self.task = task
        self.arguments = arguments
        self.conversation = conversation
        self.name = task.name

        self._out: R | NoResult = NoResult()
        current_flow = Flow.get_current()
        if current_flow is None:
            raise RuntimeError("Cannot create a node without an active flow")
        self.id = current_flow.add_node(node=self)

    @property
    def out(self) -> R:
        if isinstance(self._out, NoResult):
            raise RuntimeError("Cannot access result of a node before it has run")
        return self._out

    async def generate(self) -> None:
        self._out = await self.task.generate(**self.arguments)

    async def regenerate(self) -> None:
        if self.conversation is None:
            raise RuntimeError("Cannot regenerate a node without a conversation")

        self._out = await self.task.regenerate(
            result=self.out, conversation=self.conversation
        )

    def get_inbound_nodes(self) -> list["Node[Any]"]:
        def __find_nodes(
            obj: Any, visited: set[int] | None = None
        ) -> list["Node[Any]"]:
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


class Task[**P, R]:
    def __init__(
        self, generator: Callable[P, Awaitable[R]], /, *, retries: int = 0
    ) -> None:
        self.retries = retries
        self._generator = generator
        self._regenerator: IRegenerator[R] | None = None
        self._signature = inspect.signature(generator)
        self.__validate_fn()

    @property
    def name(self) -> str:
        return self._generator.__name__

    @property
    def regenerator(self) -> IRegenerator[R]:
        if self._regenerator is None:
            raise RuntimeError(f"No regenerator was attached to the {self.name} task")

        return self._regenerator

    async def generate(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return await self._generator(*args, **kwargs)

    async def regenerate(self, *, result: R, conversation: Conversation[R]) -> R:
        return await self.regenerator(result=result, conversation=conversation)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Node[R]:
        arguments = self._signature.bind(*args, **kwargs).arguments
        return Node(task=self, arguments=arguments)

    def __validate_fn(self) -> None:
        if any(
            param.kind == inspect.Parameter.POSITIONAL_ONLY
            for param in self._signature.parameters.values()
        ):
            raise ValueError("Positional-only parameters are not supported in Tasks")


class Flow:
    _current: contextvars.ContextVar["Flow | None"] = contextvars.ContextVar(
        "current_flow", default=None
    )

    def __init__(self, *, name: str) -> None:
        self.name = name
        self._nodes: list[Node[Any]] = []

    def add_node(self, *, node: Node[Any]) -> str:
        existing_names = [node.name for node in self._nodes]
        number = sum(1 for name in existing_names if name == node.name)
        self._nodes.append(node)
        node_id = f"{node.name}_{number}"
        return node_id

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

    @classmethod
    def get_current(cls) -> "Flow | None":
        return cls._current.get()

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
def task[**P, R](fn: Callable[P, Awaitable[R]], /) -> Task[P, R]: ...
@overload
def task[**P, R](
    *, retries: int = 0
) -> Callable[[Callable[P, Awaitable[R]]], Task[P, R]]: ...
def task[**P, R](
    fn: Callable[P, Awaitable[R]] | None = None, /, *, retries: int = 0
) -> Task[P, R] | Callable[[Callable[P, Awaitable[R]]], Task[P, R]]:
    if fn is None:
        return lambda fn: Task(fn, retries=retries)
    return Task(fn, retries=retries)
