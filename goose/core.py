from contextlib import contextmanager
from contextvars import ContextVar
from typing import Awaitable, Callable, Iterator, Protocol, Self, cast, overload

from pydantic import BaseModel

from goose.errors import Honk
from goose.types.conversation import ConversationState, GooseResponse, Result


class IAdapter[ResultT: Result](Protocol):
    async def __call__(
        self, *, conversation_state: ConversationState[ResultT]
    ) -> ResultT: ...


class TaskRunContext[ResultT: Result](BaseModel):
    task_name: str
    conversation_state: ConversationState[ResultT]
    last_input_hash: int


class FlowRunContext:
    def __init__(self, tasks: list[TaskRunContext[Result]]) -> None:
        self._tasks = tasks

    def get[R: Result](
        self, *, task_name: str, result_type: type[R]
    ) -> TaskRunContext[R]:
        for task in self._tasks:
            if task.task_name == task_name:
                return cast(TaskRunContext[R], task)

        return TaskRunContext(
            task_name=task_name,
            conversation_state=ConversationState[R](messages=[]),
            last_input_hash=0,
        )


_current_flow_run_context: ContextVar[FlowRunContext | None] = ContextVar(
    "current_flow_run_context", default=None
)


class Flow[**P, ResultT: Result]:
    def __init__(
        self, fn: Callable[P, Awaitable[ResultT]], /, *, name: str | None = None
    ) -> None:
        self._fn = fn
        self._name = name

    @property
    def name(self) -> str:
        return self._name or self._fn.__name__

    @contextmanager
    def run(self, ctx: FlowRunContext) -> Iterator[None]:
        old_ctx = _current_flow_run_context.get()
        _current_flow_run_context.set(ctx)
        try:
            yield
        finally:
            _current_flow_run_context.set(old_ctx)

    async def generate(self, *args: P.args, **kwargs: P.kwargs) -> ResultT:
        result = await self._fn(*args, **kwargs)
        return result


class Task[**P, R: Result]:
    def __init__(
        self,
        generator: Callable[P, Awaitable[R]],
        /,
        *,
        retries: int = 0,
        name: str | None = None,
    ) -> None:
        self._generator = generator
        self._adapter: IAdapter[R] | None = None
        self._name = name
        self._retries = retries

    @property
    def result_type(self) -> type[R]:
        result_type = self._generator.__annotations__.get("return")
        if result_type is None:
            raise Honk(f"Task {self.name} has no return type annotation")
        return result_type

    @property
    def name(self) -> str:
        return self._name or self._generator.__name__

    def adapter(self, adapter: IAdapter[R]) -> Self:
        self._adapter = adapter
        return self

    async def run(
        self, run_context: TaskRunContext[R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        input_hash = self.__hash_input(*args, **kwargs)
        if input_hash != run_context.last_input_hash:
            result = await self._generator(*args, **kwargs)
            run_context.conversation_state.messages[-1] = GooseResponse(result=result)
            run_context.last_input_hash = input_hash
            return result
        elif run_context.conversation_state.awaiting_response:
            if self._adapter is None:
                raise Honk("No adapter provided for Task")
            result = await self._adapter(
                conversation_state=run_context.conversation_state
            )
            run_context.conversation_state.messages.append(GooseResponse(result=result))
            return result
        else:
            if not isinstance(
                run_context.conversation_state.messages[-1], GooseResponse
            ):
                raise Honk(
                    "Conversation must alternate between User and Result messages"
                )
            return run_context.conversation_state.messages[-1].result

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        flow_context = self.__get_current_flow_run_context()
        task_context = flow_context.get(
            task_name=self.name, result_type=self.result_type
        )
        return await self.run(task_context, *args, **kwargs)

    def __hash_input(self, *args: P.args, **kwargs: P.kwargs) -> int:
        return hash(tuple(args) + tuple(kwargs.values()))

    def __get_current_flow_run_context(self) -> FlowRunContext:
        ctx = _current_flow_run_context.get()
        if ctx is None:
            raise Honk("No current flow run context")
        return ctx


@overload
def task[**P, R: Result](generator: Callable[P, Awaitable[R]], /) -> Task[P, R]: ...
@overload
def task[**P, R: Result](
    *, retries: int = 0, name: str | None = None
) -> Callable[[Callable[P, Awaitable[R]]], Task[P, R]]: ...
def task[**P, R: Result](
    generator: Callable[P, Awaitable[R]] | None = None,
    /,
    *,
    retries: int = 0,
    name: str | None = None,
) -> Task[P, R] | Callable[[Callable[P, Awaitable[R]]], Task[P, R]]:
    if generator is None:

        def decorator(fn: Callable[P, Awaitable[R]]) -> Task[P, R]:
            return Task(fn, retries=retries, name=name)

        return decorator

    return Task(generator, retries=retries, name=name)


@overload
def flow[**P, R: Result](fn: Callable[P, Awaitable[R]], /) -> Flow[P, R]: ...
@overload
def flow[**P, R: Result](
    *, name: str | None = None
) -> Callable[[Callable[P, Awaitable[R]]], Flow[P, R]]: ...
def flow[**P, R: Result](
    fn: Callable[P, Awaitable[R]] | None = None, /, *, name: str | None = None
) -> Flow[P, R] | Callable[[Callable[P, Awaitable[R]]], Flow[P, R]]:
    if fn is None:

        def decorator(fn: Callable[P, Awaitable[R]]) -> Flow[P, R]:
            return Flow(fn, name=name)

        return decorator

    return Flow(fn, name=name)
