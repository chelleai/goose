import hashlib
from collections.abc import Awaitable, Callable
from typing import Any, overload

from aikernel import LLMModelAlias, LLMRouter, LLMSystemMessage, LLMUserMessage
from pydantic import BaseModel

from goose._internal.agent import Agent
from goose._internal.result import Result
from goose._internal.state import FlowRun, NodeState, get_current_flow_run
from goose.errors import Honk


class Task[**P, R: Result]:
    def __init__(
        self,
        generator: Callable[P, Awaitable[R]],
        /,
        *,
        retries: int = 0,
    ) -> None:
        self._generator = generator
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

    async def generate(self, state: NodeState, *args: P.args, **kwargs: P.kwargs) -> R:
        state_hash = self.__hash_task_call(*args, **kwargs)
        if state_hash != state.last_hash:
            result = await self._generator(*args, **kwargs)
            state.add_result(result=result.model_dump_json(), new_hash=state_hash, overwrite=True)
            return result
        else:
            return self.result_type.model_validate_json(state.raw_result)

    async def ask[M: LLMModelAlias](
        self,
        *,
        user_message: LLMUserMessage,
        router: LLMRouter[M],
        model: M,
        context: LLMSystemMessage | None = None,
        index: int = 0,
    ) -> str:
        flow_run = self.__get_current_flow_run()
        node_state = flow_run.get_state(task=self, index=index)

        if len(node_state.conversation.assistant_messages) == 0:
            raise Honk("Cannot ask about a task that has not been initially generated")

        if context is not None:
            node_state.set_context(context=context)
        node_state.add_user_message(message=user_message)

        answer = await flow_run.agent(
            messages=node_state.conversation.render(),
            model=model,
            task_name=f"ask--{self.name}",
            mode="ask",
            router=router,
        )
        node_state.add_answer(answer=answer)
        flow_run.upsert_node_state(node_state)

        return answer

    async def refine[M: LLMModelAlias](
        self,
        *,
        user_message: LLMUserMessage,
        router: LLMRouter[M],
        model: M,
        context: LLMSystemMessage | None = None,
        index: int = 0,
    ) -> R:
        flow_run = self.__get_current_flow_run()
        node_state = flow_run.get_state(task=self, index=index)

        if len(node_state.conversation.assistant_messages) == 0:
            raise Honk("Cannot refine a task that has not been initially generated")

        if context is not None:
            node_state.set_context(context=context)
        node_state.add_user_message(message=user_message)

        result = await flow_run.agent(
            messages=node_state.conversation.render(),
            model=model,
            task_name=f"refine--{self.name}",
            response_model=self.result_type,
            mode="refine",
            router=router,
        )
        node_state.add_result(result=result.model_dump_json())
        flow_run.upsert_node_state(node_state)

        return result

    def edit(self, *, result: R, index: int = 0) -> None:
        flow_run = self.__get_current_flow_run()
        node_state = flow_run.get_state(task=self, index=index)
        node_state.edit_last_result(result=result.model_dump_json())
        flow_run.upsert_node_state(node_state)

    def undo(self, *, index: int = 0) -> None:
        flow_run = self.__get_current_flow_run()
        node_state = flow_run.get_state(task=self, index=index)
        node_state.undo()
        flow_run.upsert_node_state(node_state)

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        flow_run = self.__get_current_flow_run()
        node_state = flow_run.get_next_state(task=self)
        result = await self.generate(node_state, *args, **kwargs)
        flow_run.upsert_node_state(node_state)
        return result

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
def task[**P, R: Result](*, retries: int = 0) -> Callable[[Callable[P, Awaitable[R]]], Task[P, R]]: ...
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
