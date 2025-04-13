"""Task module for defining and executing LLM tasks.

This module provides the Task class and task decorator for creating and executing
individual LLM tasks within a flow. Tasks handle execution, caching, conversation
management, and result refinement.
"""

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
    """A task within a flow that produces a structured result.

    Tasks are the building blocks of flows and represent individual LLM operations.
    They handle execution, result caching, conversation management, and result refinement.

    Type Parameters:
        P: Parameter types for the task function
        R: Result type for the task, must be a subclass of Result
    """

    def __init__(
        self,
        generator: Callable[P, Awaitable[R]],
        /,
        *,
        retries: int = 0,
    ) -> None:
        """Initialize a Task.

        Args:
            generator: The function that implements the task
            retries: Number of automatic retries if the task fails
        """
        self._generator = generator
        self._retries = retries

    @property
    def result_type(self) -> type[R]:
        """Get the return type of the task.

        Returns:
            The result type class for this task

        Raises:
            Honk: If the task function has no return type annotation
        """
        result_type = self._generator.__annotations__.get("return")
        if result_type is None:
            raise Honk(f"Task {self.name} has no return type annotation")
        return result_type

    @property
    def name(self) -> str:
        """Get the name of the task.

        Returns:
            The name of the task, derived from the generator function name
        """
        return self._generator.__name__

    async def generate(self, state: NodeState, *args: P.args, **kwargs: P.kwargs) -> R:
        """Generate a result for this task.

        Executes the task generator function to produce a result. Uses caching
        based on input hashing to avoid redundant executions.

        Args:
            state: The current node state for this task execution
            *args: Positional arguments for the task function
            **kwargs: Keyword arguments for the task function

        Returns:
            The generated result, either freshly computed or from cache
        """
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
        """Ask a follow-up question about a task's result.

        This method allows for a conversational interaction with a task after
        it has been executed, to ask questions or get explanations about the result.

        Args:
            user_message: The user's question or message
            router: LLM router for routing the request
            model: The model to use for generating the response
            context: Optional system message to provide context
            index: Index of the task instance when the same task appears multiple times

        Returns:
            The response text from the model

        Raises:
            Honk: If the task has not been initially generated
        """
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
        """Refine a task's result based on feedback.

        This method allows for iterative refinement of a task's result based on
        user feedback or additional requirements.

        Args:
            user_message: The user's feedback or refinement request
            router: LLM router for routing the request
            model: The model to use for generating the response
            context: Optional system message to provide context
            index: Index of the task instance when the same task appears multiple times

        Returns:
            A refined result of the same type as the original result

        Raises:
            Honk: If the task has not been initially generated
        """
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
        """Manually edit a task's result.

        This method allows for direct editing of a task's result, bypassing
        the usual generation process.

        Args:
            result: The new result to use
            index: Index of the task instance to edit
        """
        flow_run = self.__get_current_flow_run()
        node_state = flow_run.get_state(task=self, index=index)
        node_state.edit_last_result(result=result.model_dump_json())
        flow_run.upsert_node_state(node_state)

    def undo(self, *, index: int = 0) -> None:
        """Undo the most recent change to a task's state.

        This method reverts the task's state to its previous version.

        Args:
            index: Index of the task instance to undo
        """
        flow_run = self.__get_current_flow_run()
        node_state = flow_run.get_state(task=self, index=index)
        node_state.undo()
        flow_run.upsert_node_state(node_state)

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Execute the task within a flow.

        This method is called when the task is invoked as a function within a flow.
        It manages state setup, generation, and state persistence.

        Args:
            *args: Positional arguments for the task function
            **kwargs: Keyword arguments for the task function

        Returns:
            The result of the task
        """
        flow_run = self.__get_current_flow_run()
        node_state = flow_run.get_next_state(task=self)
        result = await self.generate(node_state, *args, **kwargs)
        flow_run.upsert_node_state(node_state)
        return result

    def __hash_task_call(self, *args: P.args, **kwargs: P.kwargs) -> int:
        """Create a hash of task arguments for caching.

        Generates a unique hash based on the task's input arguments
        to determine when inputs have changed and results need to be regenerated.

        Args:
            *args: Positional arguments to hash
            **kwargs: Keyword arguments to hash

        Returns:
            An integer hash of the arguments

        Raises:
            Honk: If an argument cannot be hashed
        """

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
    """Decorator for creating Task instances.

    This decorator transforms async functions into Task objects that can be
    used in flows. Tasks handle execution, caching, conversation, and result refinement.

    Examples:
        ```python
        @task
        async def generate_summary(text: str) -> TextResult:
            # Task implementation...

        # Or with parameters
        @task(retries=2)
        async def generate_summary(text: str) -> TextResult:
            # Task implementation...
        ```

    Args:
        generator: The function to decorate
        retries: Number of automatic retries if the task fails

    Returns:
        A Task instance, or a decorator function if used with parameters
    """
    if generator is None:

        def decorator(fn: Callable[P, Awaitable[R]]) -> Task[P, R]:
            return Task(fn, retries=retries)

        return decorator

    return Task(generator, retries=retries)
