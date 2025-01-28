import json
from contextlib import contextmanager
from contextvars import ContextVar
from typing import (
    Any,
    Awaitable,
    Callable,
    Iterator,
    NewType,
    Protocol,
    Self,
    overload,
)

from pydantic import BaseModel, ConfigDict, field_validator

from goose.agent import (
    Agent,
    AssistantMessage,
    IAgentLogger,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from goose.errors import Honk

SerializedFlowRun = NewType("SerializedFlowRun", str)


class Result(BaseModel):
    model_config = ConfigDict(frozen=True)


class GooseResponse[R: Result](BaseModel):
    result: R


class Conversation[R: Result](BaseModel):
    messages: list[UserMessage | GooseResponse[R]]
    context: SystemMessage | None = None

    @field_validator("messages")
    def alternates_starting_with_result(
        cls, messages: list[UserMessage | GooseResponse[R]]
    ) -> list[UserMessage | GooseResponse[R]]:
        if len(messages) == 0:
            return messages
        elif isinstance(messages[0], UserMessage):
            raise Honk(
                "User cannot start a conversation on a Task, must begin with a Result"
            )

        last_message_type: type[UserMessage | GooseResponse[R]] = type(messages[0])
        for message in messages[1:]:
            if isinstance(message, last_message_type):
                raise Honk(
                    "Conversation must alternate between User and Result messages"
                )
            last_message_type = type(message)

        return messages

    @property
    def awaiting_response(self) -> bool:
        return len(self.messages) % 2 == 0

    def render(self) -> list[LLMMessage]:
        messages: list[LLMMessage] = []
        if self.context is not None:
            messages.append(self.context.render())

        for message in self.messages:
            if isinstance(message, UserMessage):
                messages.append(message.render())
            else:
                messages.append(
                    AssistantMessage(text=message.result.model_dump_json()).render()
                )

        return messages


class IAdapter[ResultT: Result](Protocol):
    async def __call__(self, *, conversation: Conversation[ResultT]) -> ResultT: ...


class NodeState[ResultT: Result](BaseModel):
    task_name: str
    index: int
    conversation: Conversation[ResultT]
    last_input_hash: int

    @property
    def result(self) -> ResultT:
        last_message = self.conversation.messages[-1]
        if isinstance(last_message, GooseResponse):
            return last_message.result
        else:
            raise Honk("Node awaiting response, has no result")

    def set_context(self, *, context: SystemMessage) -> Self:
        self.conversation.context = context
        return self

    def add_result(
        self,
        *,
        result: ResultT,
        new_input_hash: int | None = None,
        overwrite: bool = False,
    ) -> Self:
        if overwrite:
            if len(self.conversation.messages) == 0:
                self.conversation.messages.append(GooseResponse(result=result))
            else:
                self.conversation.messages[-1] = GooseResponse(result=result)
        else:
            self.conversation.messages.append(GooseResponse(result=result))
        if new_input_hash is not None:
            self.last_input_hash = new_input_hash
        return self

    def add_user_message(self, *, message: UserMessage) -> Self:
        self.conversation.messages.append(message)
        return self


class FlowRun:
    def __init__(self) -> None:
        self._node_states: dict[tuple[str, int], str] = {}
        self._last_requested_indices: dict[str, int] = {}
        self._flow_name = ""
        self._id = ""
        self._agent: Agent | None = None

    @property
    def flow_name(self) -> str:
        return self._flow_name

    @property
    def id(self) -> str:
        return self._id

    @property
    def agent(self) -> Agent:
        if self._agent is None:
            raise Honk("Agent is only accessible once a run is started")
        return self._agent

    def add(self, node_state: NodeState[Any], /) -> None:
        key = (node_state.task_name, node_state.index)
        self._node_states[key] = node_state.model_dump_json()

    def get_next[R: Result](self, *, task: "Task[Any, R]") -> NodeState[R]:
        if task.name not in self._last_requested_indices:
            self._last_requested_indices[task.name] = 0
        else:
            self._last_requested_indices[task.name] += 1

        return self.get(task=task, index=self._last_requested_indices[task.name])

    def get_all[R: Result](self, *, task: "Task[Any, R]") -> list[NodeState[R]]:
        matching_nodes: list[NodeState[R]] = []
        for key, node_state in self._node_states.items():
            if key[0] == task.name:
                matching_nodes.append(
                    NodeState[task.result_type].model_validate_json(node_state)
                )
        return matching_nodes

    def get[R: Result](self, *, task: "Task[Any, R]", index: int = 0) -> NodeState[R]:
        if (
            existing_node_state := self._node_states.get((task.name, index))
        ) is not None:
            return NodeState[task.result_type].model_validate_json(existing_node_state)
        else:
            return NodeState[task.result_type](
                task_name=task.name,
                index=index or 0,
                conversation=Conversation[task.result_type](messages=[]),
                last_input_hash=0,
            )

    def start(
        self,
        *,
        flow_name: str,
        run_id: str,
        agent_logger: IAgentLogger | None = None,
    ) -> None:
        self._last_requested_indices = {}
        self._flow_name = flow_name
        self._id = run_id
        self._agent = Agent(
            flow_name=self.flow_name, run_id=self.id, logger=agent_logger
        )

    def end(self) -> None:
        self._last_requested_indices = {}
        self._flow_name = ""
        self._id = ""
        self._agent = None

    def dump(self) -> SerializedFlowRun:
        return SerializedFlowRun(
            json.dumps(
                {
                    ":".join([task_name, str(index)]): value
                    for (task_name, index), value in self._node_states.items()
                }
            )
        )

    @classmethod
    def load(cls, run: SerializedFlowRun, /) -> Self:
        flow_run = cls()
        raw_node_states = json.loads(run)
        new_node_states: dict[tuple[str, int], str] = {}
        for key, node_state in raw_node_states.items():
            task_name, index = tuple(key.split(":"))
            new_node_states[(task_name, int(index))] = node_state

        flow_run._node_states = new_node_states
        return flow_run


_current_flow_run: ContextVar[FlowRun | None] = ContextVar(
    "current_flow_run", default=None
)


class Flow[**P]:
    def __init__(
        self,
        fn: Callable[P, Awaitable[None]],
        /,
        *,
        name: str | None = None,
        agent_logger: IAgentLogger | None = None,
    ) -> None:
        self._fn = fn
        self._name = name
        self._agent_logger = agent_logger

    @property
    def name(self) -> str:
        return self._name or self._fn.__name__

    @property
    def current_run(self) -> FlowRun:
        run = _current_flow_run.get()
        if run is None:
            raise Honk("No current flow run")
        return run

    @contextmanager
    def start_run(
        self, *, run_id: str, preload: FlowRun | None = None
    ) -> Iterator[FlowRun]:
        if preload is None:
            run = FlowRun()
        else:
            run = preload

        old_run = _current_flow_run.get()
        _current_flow_run.set(run)

        run.start(flow_name=self.name, run_id=run_id, agent_logger=self._agent_logger)
        yield run
        run.end()

        _current_flow_run.set(old_run)

    async def generate(self, *args: P.args, **kwargs: P.kwargs) -> None:
        await self._fn(*args, **kwargs)


class Task[**P, R: Result]:
    def __init__(
        self,
        generator: Callable[P, Awaitable[R]],
        /,
        *,
        retries: int = 0,
    ) -> None:
        self._generator = generator
        self._adapter: IAdapter[R] | None = None
        self._retries = retries

    @property
    def result_type(self) -> type[R]:
        result_type = self._generator.__annotations__.get("return")
        if result_type is None:
            raise Honk(f"Task {self.name} has no return type annotation")
        return result_type

    @property
    def name(self) -> str:
        return self._generator.__name__

    def adapter(self, adapter: IAdapter[R]) -> Self:
        self._adapter = adapter
        return self

    async def generate(
        self, state: NodeState[R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        input_hash = self.__hash_input(*args, **kwargs)
        if input_hash != state.last_input_hash:
            result = await self._generator(*args, **kwargs)
            state.add_result(result=result, new_input_hash=input_hash, overwrite=True)
            return result
        else:
            if not isinstance(state.conversation.messages[-1], GooseResponse):
                raise Honk(
                    "Conversation must alternate between User and Result messages"
                )
            return state.result

    async def jam(
        self,
        *,
        user_message: UserMessage,
        context: SystemMessage | None = None,
        index: int = 0,
    ) -> R:
        flow_run = self.__get_current_flow_run()
        node_state = flow_run.get(task=self, index=index)
        if self._adapter is None:
            raise Honk("No adapter provided for Task")

        if context is not None:
            node_state.set_context(context=context)
        node_state.add_user_message(message=user_message)

        result = await self._adapter(conversation=node_state.conversation)
        node_state.add_result(result=result)
        flow_run.add(node_state)

        return result

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        flow_run = self.__get_current_flow_run()
        node_state = flow_run.get_next(task=self)
        result = await self.generate(node_state, *args, **kwargs)
        flow_run.add(node_state)
        return result

    def __hash_input(self, *args: P.args, **kwargs: P.kwargs) -> int:
        try:
            to_hash = str(tuple(args) + tuple(kwargs.values()))
            return hash(to_hash)
        except TypeError:
            raise Honk(f"Unhashable argument to task {self.name}: {args} {kwargs}")

    def __get_current_flow_run(self) -> FlowRun:
        run = _current_flow_run.get()
        if run is None:
            raise Honk("No current flow run")
        return run


@overload
def task[**P, R: Result](generator: Callable[P, Awaitable[R]], /) -> Task[P, R]: ...
@overload
def task[**P, R: Result](
    *, retries: int = 0
) -> Callable[[Callable[P, Awaitable[R]]], Task[P, R]]: ...
def task[**P, R: Result](
    generator: Callable[P, Awaitable[R]] | None = None,
    /,
    *,
    retries: int = 0,
) -> Task[P, R] | Callable[[Callable[P, Awaitable[R]]], Task[P, R]]:
    if generator is None:

        def decorator(fn: Callable[P, Awaitable[R]]) -> Task[P, R]:
            return Task(fn, retries=retries)

        return decorator

    return Task(generator, retries=retries)


@overload
def flow[**P](fn: Callable[P, Awaitable[None]], /) -> Flow[P]: ...
@overload
def flow[**P](
    *, name: str | None = None, agent_logger: IAgentLogger | None = None
) -> Callable[[Callable[P, Awaitable[None]]], Flow[P]]: ...
def flow[**P](
    fn: Callable[P, Awaitable[None]] | None = None,
    /,
    *,
    name: str | None = None,
    agent_logger: IAgentLogger | None = None,
) -> Flow[P] | Callable[[Callable[P, Awaitable[None]]], Flow[P]]:
    if fn is None:

        def decorator(fn: Callable[P, Awaitable[None]]) -> Flow[P]:
            return Flow(fn, name=name, agent_logger=agent_logger)

        return decorator

    return Flow(fn, name=name, agent_logger=agent_logger)
