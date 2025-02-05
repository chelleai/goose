import hashlib
from collections.abc import Awaitable, Callable
from typing import Any, overload

from pydantic import BaseModel

from ..errors import Honk
from .agent import Agent, GeminiModel, SystemMessage, UserMessage
from .conversation import Conversation
from .result import Result, TextResult
from .state import FlowRun, NodeState, get_current_flow_run
from .types.agent import AssistantMessage


class Task[**P, R: Result]:
    def __init__(
        self,
        generator: Callable[P, Awaitable[R]],
        /,
        *,
        retries: int = 0,
        adapter_model: GeminiModel = GeminiModel.FLASH,
    ) -> None:
        self._generator = generator
        self._retries = retries
        self._adapter_model = adapter_model
        self._adapter_model = adapter_model

    @property
    def result_type(self) -> type[R]:
        result_type = self._generator.__annotations__.get("return")
        if result_type is None:
            raise Honk(f"Task {self.name} has no return type annotation")
        return result_type

    @property
    def name(self) -> str:
        return self._generator.__name__

    async def generate(self, state: NodeState[R], *args: P.args, **kwargs: P.kwargs) -> R:
        state_hash = self.__hash_task_call(*args, **kwargs)
        if state_hash != state.last_hash:
            result = await self._generator(*args, **kwargs)
            state.add_result(result=result, new_hash=state_hash, overwrite=True)
            return result
        else:
            return state.result

    async def refine(
        self,
        *,
        user_message: UserMessage,
        context: SystemMessage | None = None,
        index: int = 0,
    ) -> R:
        flow_run = self.__get_current_flow_run()
        node_state = flow_run.get(task=self, index=index)

        if context is not None:
            node_state.set_context(context=context)
        node_state.add_user_message(message=user_message)

        result = await self.__adapt(conversation=node_state.conversation, agent=flow_run.agent)
        node_state.add_result(result=result)
        flow_run.upsert_node_state(node_state)

        return result

    def edit(self, *, result: R, index: int = 0) -> None:
        flow_run = self.__get_current_flow_run()
        node_state = flow_run.get(task=self, index=index)
        node_state.edit_last_result(result=result)
        flow_run.upsert_node_state(node_state)

    def undo(self, *, index: int = 0) -> None:
        flow_run = self.__get_current_flow_run()
        node_state = flow_run.get(task=self, index=index)
        node_state.undo()
        flow_run.upsert_node_state(node_state)

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        flow_run = self.__get_current_flow_run()
        node_state = flow_run.get_next(task=self)
        result = await self.generate(node_state, *args, **kwargs)
        flow_run.upsert_node_state(node_state)
        return result

    async def __adapt(self, *, conversation: Conversation[R], agent: Agent) -> R:
        messages: list[UserMessage | AssistantMessage] = []
        for message_index in range(len(conversation.user_messages)):
            user_message = conversation.user_messages[message_index]
            result = conversation.result_messages[message_index]

            if isinstance(result, TextResult):
                assistant_text = result.text
            else:
                assistant_text = result.model_dump_json()
            assistant_message = AssistantMessage(text=assistant_text)
            messages.append(assistant_message)
            messages.append(user_message)

        return await agent(
            messages=messages,
            model=self._adapter_model,
            task_name=f"adapt--{self.name}",
            system=conversation.context,
            response_model=self.result_type,
        )

    def __hash_task_call(self, *args: P.args, **kwargs: P.kwargs) -> int:
        def update_hash(argument: Any, current_hash: Any = hashlib.sha256()) -> None:
            try:
                if isinstance(argument, list | tuple | set):
                    for item in argument:
                        update_hash(item, current_hash)
                elif isinstance(argument, dict):
                    for key, value in argument.items():
                        update_hash(key, current_hash)
                        update_hash(value, current_hash)
                elif isinstance(argument, BaseModel):
                    update_hash(argument.model_dump_json())
                elif isinstance(argument, bytes):
                    current_hash.update(argument)
                elif isinstance(argument, Agent):
                    current_hash.update(b"AGENT")
                else:
                    current_hash.update(str(argument).encode())
            except TypeError:
                raise Honk(f"Unhashable argument to task {self.name}: {argument}")

        result = hashlib.sha256()
        update_hash(args, result)
        update_hash(kwargs, result)

        return int(result.hexdigest(), 16)

    def __get_current_flow_run(self) -> FlowRun[Any]:
        run = get_current_flow_run()
        if run is None:
            raise Honk("No current flow run")
        return run


@overload
def task[**P, R: Result](generator: Callable[P, Awaitable[R]], /) -> Task[P, R]: ...
@overload
def task[**P, R: Result](
    *, retries: int = 0, adapter_model: GeminiModel = GeminiModel.FLASH
) -> Callable[[Callable[P, Awaitable[R]]], Task[P, R]]: ...
def task[**P, R: Result](
    generator: Callable[P, Awaitable[R]] | None = None,
    /,
    *,
    retries: int = 0,
    adapter_model: GeminiModel = GeminiModel.FLASH,
) -> Task[P, R] | Callable[[Callable[P, Awaitable[R]]], Task[P, R]]:
    if generator is None:

        def decorator(fn: Callable[P, Awaitable[R]]) -> Task[P, R]:
            return Task(fn, retries=retries, adapter_model=adapter_model)

        return decorator

    return Task(generator, retries=retries, adapter_model=adapter_model)
