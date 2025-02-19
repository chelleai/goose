import json
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, NewType, Self

from pydantic import BaseModel, ConfigDict

from ..errors import Honk
from .agent import (
    Agent,
    IAgentLogger,
    SystemMessage,
    UserMessage,
)
from .conversation import Conversation
from .result import Result

if TYPE_CHECKING:
    from goose._internal.task import Task

SerializedFlowRun = NewType("SerializedFlowRun", str)


class FlowArguments(BaseModel):
    model_config = ConfigDict(frozen=True)


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

    def edit_last_result(self, *, result: ResultT) -> Self:
        if len(self.conversation.result_messages) == 0:
            raise Honk("Node awaiting response, has no result")

        self.conversation.result_messages[-1] = result
        return self

    def undo(self) -> Self:
        self.conversation.undo()
        return self


class FlowRun[FlowArgumentsT: FlowArguments]:
    def __init__(self, *, flow_arguments_model: type[FlowArgumentsT]) -> None:
        self._node_states: dict[tuple[str, int], str] = {}
        self._last_requested_indices: dict[str, int] = {}
        self._flow_name = ""
        self._id = ""
        self._agent: Agent | None = None
        self._flow_arguments: FlowArgumentsT | None = None
        self._flow_arguments_model = flow_arguments_model

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
    def flow_arguments(self) -> FlowArgumentsT:
        if self._flow_arguments is None:
            raise Honk("This Flow run has not been executed before")

        return self._flow_arguments

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

    def set_flow_arguments(self, flow_arguments: FlowArgumentsT, /) -> None:
        self._flow_arguments = flow_arguments

    def upsert_node_state(self, node_state: NodeState[Any], /) -> None:
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

    def dump(self) -> SerializedFlowRun:
        formatted_node_states = {f"{k[0]},{k[1]}": v for k, v in self._node_states.items()}
        return SerializedFlowRun(
            json.dumps({"node_states": formatted_node_states, "flow_arguments": self.flow_arguments.model_dump()})
        )

    @classmethod
    def load[T: FlowArguments](
        cls, *, serialized_flow_run: SerializedFlowRun, flow_arguments_model: type[T]
    ) -> "FlowRun[T]":
        flow_run_state = json.loads(serialized_flow_run)
        raw_node_states = flow_run_state["node_states"]
        node_states: dict[tuple[str, int], str] = {}
        for key, value in raw_node_states.items():
            task_name, index = key.split(",")
            node_states[(task_name, int(index))] = value
        flow_arguments = flow_arguments_model.model_validate(flow_run_state["flow_arguments"])

        flow_run = FlowRun(flow_arguments_model=flow_arguments_model)
        flow_run._node_states = node_states
        flow_run._flow_arguments = flow_arguments

        return flow_run


_current_flow_run: ContextVar[FlowRun[Any] | None] = ContextVar("current_flow_run", default=None)


def get_current_flow_run() -> FlowRun[Any] | None:
    return _current_flow_run.get()


def set_current_flow_run(flow_run: FlowRun[Any] | None) -> None:
    _current_flow_run.set(flow_run)
