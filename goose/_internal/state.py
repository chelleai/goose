from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel

from goose._internal.agent import (
    Agent,
    IAgentLogger,
    SystemMessage,
    UserMessage,
)
from goose._internal.conversation import Conversation
from goose._internal.result import Result
from goose.errors import Honk

if TYPE_CHECKING:
    from goose._internal.task import Task


@dataclass
class FlowRunState:
    node_states: dict[tuple[str, int], str]
    flow_args: tuple[Any, ...]
    flow_kwargs: dict[str, Any]


class NodeState[ResultT: Result](BaseModel):
    task_name: str
    index: int
    conversation: Conversation[ResultT]
    last_hash: int

    @property
    def result(self) -> ResultT:
        if len(self.conversation.result_messages) == 0:
            raise Honk("Node awaiting response, has no result")

        return self.conversation.result_messages[-1]

    def set_context(self, *, context: SystemMessage) -> Self:
        self.conversation.context = context
        return self

    def add_result(
        self,
        *,
        result: ResultT,
        new_hash: int | None = None,
        overwrite: bool = False,
    ) -> Self:
        if overwrite and len(self.conversation.result_messages) > 0:
            self.conversation.result_messages[-1] = result
        else:
            self.conversation.result_messages.append(result)
        if new_hash is not None:
            self.last_hash = new_hash
        return self

    def add_user_message(self, *, message: UserMessage) -> Self:
        self.conversation.user_messages.append(message)
        return self

    def undo(self) -> Self:
        self.conversation.undo()
        return self


class FlowRun:
    def __init__(self) -> None:
        self._node_states: dict[tuple[str, int], str] = {}
        self._last_requested_indices: dict[str, int] = {}
        self._flow_name = ""
        self._id = ""
        self._agent: Agent | None = None
        self._flow_args: tuple[Any, ...] | None = None
        self._flow_kwargs: dict[str, Any] | None = None

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

    @property
    def flow_inputs(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if self._flow_args is None or self._flow_kwargs is None:
            raise Honk("This Flow run has not been executed before")

        return self._flow_args, self._flow_kwargs

    def get_all[R: Result](self, *, task: "Task[Any, R]") -> list[NodeState[R]]:
        matching_nodes: list[NodeState[R]] = []
        for key, node_state in self._node_states.items():
            if key[0] == task.name:
                matching_nodes.append(NodeState[task.result_type].model_validate_json(node_state))
        return sorted(matching_nodes, key=lambda node: node.index)

    def get[R: Result](self, *, task: "Task[Any, R]", index: int = 0) -> NodeState[R]:
        if (existing_node_state := self._node_states.get((task.name, index))) is not None:
            return NodeState[task.result_type].model_validate_json(existing_node_state)
        else:
            return NodeState[task.result_type](
                task_name=task.name,
                index=index,
                conversation=Conversation[task.result_type](user_messages=[], result_messages=[]),
                last_hash=0,
            )

    def set_flow_inputs(self, *args: Any, **kwargs: Any) -> None:
        self._flow_args = args
        self._flow_kwargs = kwargs

    def add_node_state(self, node_state: NodeState[Any], /) -> None:
        key = (node_state.task_name, node_state.index)
        self._node_states[key] = node_state.model_dump_json()

    def get_next[R: Result](self, *, task: "Task[Any, R]") -> NodeState[R]:
        if task.name not in self._last_requested_indices:
            self._last_requested_indices[task.name] = 0
        else:
            self._last_requested_indices[task.name] += 1

        return self.get(task=task, index=self._last_requested_indices[task.name])

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
        self._agent = Agent(flow_name=self.flow_name, run_id=self.id, logger=agent_logger)

    def end(self) -> None:
        self._last_requested_indices = {}
        self._flow_name = ""
        self._id = ""
        self._agent = None

    def clear_node(self, *, task: "Task[Any, Result]", index: int) -> None:
        key = (task.name, index)
        if key in self._node_states:
            del self._node_states[key]

    def dump(self) -> FlowRunState:
        flow_args, flow_kwargs = self.flow_inputs

        return FlowRunState(
            node_states=self._node_states,
            flow_args=flow_args,
            flow_kwargs=flow_kwargs,
        )

    @classmethod
    def load(cls, flow_run_state: FlowRunState, /) -> Self:
        flow_run = cls()
        flow_run._node_states = flow_run_state.node_states
        flow_run._flow_args = flow_run_state.flow_args
        flow_run._flow_kwargs = flow_run_state.flow_kwargs

        return flow_run


_current_flow_run: ContextVar[FlowRun | None] = ContextVar("current_flow_run", default=None)


def get_current_flow_run() -> FlowRun | None:
    return _current_flow_run.get()


def set_current_flow_run(flow_run: FlowRun | None) -> None:
    _current_flow_run.set(flow_run)
