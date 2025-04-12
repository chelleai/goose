"""Agent module for interacting with language models.

This module provides the Agent class for handling interactions with language models,
along with protocols for custom logging.
"""

import logging
from datetime import datetime
from typing import Any, Literal, Protocol, overload

from aikernel import (
    LLMAssistantMessage,
    LLMModelAlias,
    LLMRouter,
    LLMSystemMessage,
    LLMToolMessage,
    LLMUserMessage,
    llm_structured,
    llm_unstructured,
)
from pydantic import ValidationError

from goose._internal.result import FindReplaceResponse, Result, TextResult
from goose._internal.types.telemetry import AgentResponse
from goose.errors import Honk

ExpectedMessage = LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage


class IAgentLogger(Protocol):
    """Protocol for custom agent response logging.

    Implement this protocol to create custom loggers for agent responses.
    """

    async def __call__(self, *, response: AgentResponse[Any]) -> None: ...


class Agent:
    """Agent for interacting with language models.

    The Agent class handles interactions with language models, including generating
    structured and unstructured responses, asking questions, and refining results.
    It also manages logging of model interactions.

    Attributes:
        flow_name: The name of the flow this agent is part of
        run_id: The ID of the current run
        logger: Optional custom logger for agent responses
    """

    def __init__(
        self,
        *,
        flow_name: str,
        run_id: str,
        logger: IAgentLogger | None = None,
    ) -> None:
        """Initialize an Agent.

        Args:
            flow_name: The name of the flow this agent is part of
            run_id: The ID of the current run
            logger: Optional custom logger for agent responses
        """
        self.flow_name = flow_name
        self.run_id = run_id
        self.logger = logger

    async def generate[R: Result, M: LLMModelAlias](
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        model: M,
        task_name: str,
        router: LLMRouter[M],
        response_model: type[R] = TextResult,
    ) -> R:
        """Generate a structured response from the language model.

        This method sends a sequence of messages to the language model and expects
        a structured response conforming to the provided response_model.

        Args:
            messages: List of messages to send to the language model
            model: The language model alias to use
            task_name: Name of the task for logging and tracking
            router: LLM router for routing the request
            response_model: Pydantic model class for the expected response structure

        Returns:
            A validated instance of the response_model

        Raises:
            ValidationError: If the response cannot be parsed into the response_model
        """
        start_time = datetime.now()
        typed_messages: list[ExpectedMessage] = [*messages]

        if response_model is TextResult:
            response = await llm_unstructured(messages=typed_messages, router=router)
            parsed_response = response_model.model_validate({"text": response.text})
        else:
            response = await llm_structured(
                messages=typed_messages, response_model=response_model, router=router
            )
            parsed_response = response.structured_response

        end_time = datetime.now()

        if isinstance(messages[0], LLMSystemMessage):
            system = messages[0].render()
            input_messages = [message.render() for message in messages[1:]]
        else:
            system = None
            input_messages = [message.render() for message in messages]

        agent_response = AgentResponse(
            response=parsed_response,
            run_id=self.run_id,
            flow_name=self.flow_name,
            task_name=task_name,
            model=model,
            system=system,
            input_messages=input_messages,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            start_time=start_time,
            end_time=end_time,
        )

        if self.logger is not None:
            await self.logger(response=agent_response)
        else:
            logging.info(agent_response.model_dump())

        return parsed_response

    async def ask[M: LLMModelAlias](
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        model: M,
        task_name: str,
        router: LLMRouter[M],
    ) -> str:
        """Ask the language model for an unstructured text response.

        This method sends a sequence of messages to the language model and
        receives a free-form text response.

        Args:
            messages: List of messages to send to the language model
            model: The language model alias to use
            task_name: Name of the task for logging and tracking
            router: LLM router for routing the request

        Returns:
            The text response from the language model
        """
        start_time = datetime.now()
        typed_messages: list[ExpectedMessage] = [*messages]
        response = await llm_unstructured(messages=typed_messages, router=router)
        end_time = datetime.now()

        if isinstance(messages[0], LLMSystemMessage):
            system = messages[0].render()
            input_messages = [message.render() for message in messages[1:]]
        else:
            system = None
            input_messages = [message.render() for message in messages]

        agent_response = AgentResponse(
            response=response.text,
            run_id=self.run_id,
            flow_name=self.flow_name,
            task_name=task_name,
            model=model,
            system=system,
            input_messages=input_messages,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            start_time=start_time,
            end_time=end_time,
        )

        if self.logger is not None:
            await self.logger(response=agent_response)
        else:
            logging.info(agent_response.model_dump())

        return response.text

    async def refine[R: Result, M: LLMModelAlias](
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        model: M,
        router: LLMRouter[M],
        task_name: str,
        response_model: type[R],
    ) -> R:
        """Refine a previous structured response based on feedback.

        This method uses a find-and-replace approach to refine a previous structured
        response. It identifies parts of the previous response to change and applies
        these changes to create an updated response.

        Args:
            messages: List of messages including the previous response and feedback
            model: The language model alias to use
            router: LLM router for routing the request
            task_name: Name of the task for logging and tracking
            response_model: The model class of the response to refine

        Returns:
            A refined instance of the response_model

        Raises:
            Honk: If no previous result is found in the message history
            ValidationError: If the refined result cannot be validated against the response_model
        """
        start_time = datetime.now()
        typed_messages: list[ExpectedMessage] = [*messages]
        find_replace_response = await llm_structured(
            model=model, messages=typed_messages, response_model=FindReplaceResponse, router=router
        )
        parsed_find_replace_response = find_replace_response.structured_response
        end_time = datetime.now()

        if isinstance(messages[0], LLMSystemMessage):
            system = messages[0].render()
            input_messages = [message.render() for message in messages[1:]]
        else:
            system = None
            input_messages = [message.render() for message in messages]

        agent_response = AgentResponse(
            response=parsed_find_replace_response,
            run_id=self.run_id,
            flow_name=self.flow_name,
            task_name=task_name,
            model=model,
            system=system,
            input_messages=input_messages,
            input_tokens=find_replace_response.usage.input_tokens,
            output_tokens=find_replace_response.usage.output_tokens,
            start_time=start_time,
            end_time=end_time,
        )

        if self.logger is not None:
            await self.logger(response=agent_response)
        else:
            logging.info(agent_response.model_dump())

        refined_response = self.__apply_find_replace(
            result=self.__find_last_result(messages=messages, response_model=response_model),
            find_replace_response=parsed_find_replace_response,
            response_model=response_model,
        )

        return refined_response

    @overload
    async def __call__[R: Result, M: LLMModelAlias](
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        model: M,
        router: LLMRouter[M],
        task_name: str,
        mode: Literal["generate"],
        response_model: type[R],
    ) -> R: ...

    @overload
    async def __call__[R: Result, M: LLMModelAlias](
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        model: M,
        router: LLMRouter[M],
        task_name: str,
        mode: Literal["ask"],
        response_model: type[R] = TextResult,
    ) -> str: ...

    @overload
    async def __call__[R: Result, M: LLMModelAlias](
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        model: M,
        router: LLMRouter[M],
        task_name: str,
        response_model: type[R],
        mode: Literal["refine"],
    ) -> R: ...

    @overload
    async def __call__[R: Result, M: LLMModelAlias](
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        model: M,
        router: LLMRouter[M],
        task_name: str,
        response_model: type[R],
    ) -> R: ...

    async def __call__[R: Result, M: LLMModelAlias](
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        model: M,
        router: LLMRouter[M],
        task_name: str,
        response_model: type[R] = TextResult,
        mode: Literal["generate", "ask", "refine"] = "generate",
    ) -> R | str:
        match mode:
            case "generate":
                return await self.generate(
                    messages=messages, model=model, task_name=task_name, router=router, response_model=response_model
                )
            case "ask":
                return await self.ask(messages=messages, model=model, task_name=task_name, router=router)
            case "refine":
                return await self.refine(
                    messages=messages, model=model, task_name=task_name, router=router, response_model=response_model
                )

    def __apply_find_replace[R: Result](
        self, *, result: R, find_replace_response: FindReplaceResponse, response_model: type[R]
    ) -> R:
        """Apply find-replace operations to a result.

        Takes a result object and a set of replacements, applies the replacements,
        and validates the new result against the response model.

        Args:
            result: The original result to modify
            find_replace_response: Object containing the replacements to apply
            response_model: The model class to validate the result against

        Returns:
            A new instance of the response model with replacements applied
        """
        dumped_result = result.model_dump_json()
        for replacement in find_replace_response.replacements:
            dumped_result = dumped_result.replace(replacement.find, replacement.replace)

        return response_model.model_validate_json(dumped_result)

    def __find_last_result[R: Result](
        self, *, messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage], response_model: type[R]
    ) -> R:
        """Find the last result in a conversation history.

        Searches through messages in reverse order to find the most recent
        assistant message that can be parsed as the given response model.

        Args:
            messages: List of messages to search through
            response_model: The model class to validate found results against

        Returns:
            The last result that can be validated as the response model

        Raises:
            Honk: If no valid result is found in the message history
        """
        for message in reversed(messages):
            if isinstance(message, LLMAssistantMessage):
                try:
                    return response_model.model_validate_json(message.parts[0].content)
                except ValidationError:
                    continue

        raise Honk("No last result found, failed to refine")
