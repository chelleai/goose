import json
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, NewType, Self

from aikernel import LLMAssistantMessage, LLMSystemMessage, LLMUserMessage
from pydantic import BaseModel, ConfigDict

from goose._internal.agent import Agent, IAgentLogger
from goose._internal.conversation import Conversation
from goose._internal.result import Result
from goose.errors import Honk

if TYPE_CHECKING:
    from goose._internal.task import Task

SerializedFlowRun = NewType("SerializedFlowRun", str)


class FlowArguments(BaseModel):
    model_config = ConfigDict(frozen=True)


class NodeState(BaseModel):
    task_name: str
    index: int
    conversation: Conversation
    last_hash: int

    @property
    def raw_result(self) -> str:
        for message in reversed(self.conversation.assistant_messages):
            if self.__message_is_result(message):
                return message.parts[0].content

        raise Honk("Node awaiting response, has no result")

    def set_context(self, *, context: LLMSystemMessage) -> Self:
        self.conversation.context = context
        return self

    def add_result(
        self,
        *,
        result: str,
        new_hash: int | None = None,
        overwrite: bool = False,
    ) -> Self:
        if overwrite and len(self.conversation.assistant_messages) > 0:
            self.conversation.assistant_messages[-1] = LLMAssistantMessage.from_text(result)
        else:
            self.conversation.assistant_messages.append(LLMAssistantMessage.from_text(result))
        if new_hash is not None:
            self.last_hash = new_hash
        return self

    def add_answer(self, *, answer: str) -> Self:
        self.conversation.assistant_messages.append(LLMAssistantMessage.from_text(answer))
        return self

    def add_user_message(self, *, message: LLMUserMessage) -> Self:
        self.conversation.user_messages.append(message)
        return self

    def edit_last_result(self, *, result: str) -> Self:
        if len(self.conversation.assistant_messages) == 0:
            raise Honk("Node awaiting response, has no result")

        for message_index, message in enumerate(reversed(self.conversation.assistant_messages)):
            if self.__message_is_result(message):
                index = len(self.conversation.assistant_messages) - message_index - 1
                self.conversation.assistant_messages[index] = LLMAssistantMessage.from_text(result)
                return self

        raise Honk("Node awaiting response, has no result")

    def undo(self) -> Self:
        self.conversation.undo()
        return self

    def __message_is_result(self, message: LLMAssistantMessage, /) -> bool:
        try:
            _ = json.loads(message.parts[0].content)
            return True
        except json.JSONDecodeError:
            return False


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

    def get_state(self, *, task: "Task[Any, Any]", index: int = 0) -> NodeState:
        if (existing_node_state := self._node_states.get((task.name, index))) is not None:
            return NodeState.model_validate_json(existing_node_state)
        else:
            return NodeState(
                task_name=task.name,
                index=index,
                conversation=Conversation(user_messages=[], assistant_messages=[]),
                last_hash=0,
            )

    def get_next_state(self, *, task: "Task[Any, Any]", index: int = 0) -> NodeState:
        if task.name not in self._last_requested_indices:
            self._last_requested_indices[task.name] = 0
        else:
            self._last_requested_indices[task.name] += 1

        return self.get_state(task=task, index=self._last_requested_indices[task.name])

    def get_all_results[R: Result](self, *, task: "Task[Any, R]") -> list[R]:
        matching_nodes: list[NodeState] = []
        for key, node_state in self._node_states.items():
            if key[0] == task.name:
                matching_nodes.append(NodeState.model_validate_json(node_state))

        sorted_nodes = sorted(matching_nodes, key=lambda node: node.index)
        return [task.result_type.model_validate_json(node.raw_result) for node in sorted_nodes]

    def get_result[R: Result](self, *, task: "Task[Any, R]", index: int = 0) -> R:
        if (existing_node_state := self._node_states.get((task.name, index))) is not None:
            parsed_node_state = NodeState.model_validate_json(existing_node_state)
            return task.result_type.model_validate_json(parsed_node_state.raw_result)
        else:
            raise Honk(f"No result found for task {task.name} at index {index}")

    def set_flow_arguments(self, flow_arguments: FlowArgumentsT, /) -> None:
        self._flow_arguments = flow_arguments

    def upsert_node_state(self, node_state: NodeState, /) -> None:
        key = (node_state.task_name, node_state.index)
        self._node_states[key] = node_state.model_dump_json()

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
