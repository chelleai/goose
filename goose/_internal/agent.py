import logging
from datetime import datetime
from typing import Any, Literal, Protocol, overload

from aikernel import (
    LLMAssistantMessage,
    LLMModelAlias,
    LLMSystemMessage,
    LLMToolMessage,
    LLMUserMessage,
    Router,
    llm_structured,
    llm_unstructured,
)
from pydantic import ValidationError

from goose._internal.result import FindReplaceResponse, Result, TextResult
from goose._internal.types.telemetry import AgentResponse
from goose.errors import Honk

ExpectedMessage = LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage


class IAgentLogger(Protocol):
    async def __call__(self, *, response: AgentResponse[Any]) -> None: ...


class Agent:
    def __init__(
        self,
        *,
        flow_name: str,
        run_id: str,
        logger: IAgentLogger | None = None,
    ) -> None:
        self.flow_name = flow_name
        self.run_id = run_id
        self.logger = logger

    async def generate[R: Result](
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        model: LLMModelAlias,
        task_name: str,
        router: Router,
        response_model: type[R] = TextResult,
    ) -> R:
        start_time = datetime.now()
        typed_messages: list[ExpectedMessage] = [*messages]

        if response_model is TextResult:
            response = await llm_unstructured(model=model, messages=typed_messages, router=router)
            parsed_response = response_model.model_validate({"text": response.text})
        else:
            response = await llm_structured(
                model=model, messages=typed_messages, response_model=response_model, router=router
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

    async def ask(
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        model: LLMModelAlias,
        task_name: str,
        router: Router,
    ) -> str:
        start_time = datetime.now()
        typed_messages: list[ExpectedMessage] = [*messages]
        response = await llm_unstructured(model=model, messages=typed_messages, router=router)
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

    async def refine[R: Result](
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        model: LLMModelAlias,
        router: Router,
        task_name: str,
        response_model: type[R],
    ) -> R:
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
    async def __call__[R: Result](
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        model: LLMModelAlias,
        router: Router,
        task_name: str,
        mode: Literal["generate"],
        response_model: type[R],
    ) -> R: ...

    @overload
    async def __call__[R: Result](
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        model: LLMModelAlias,
        router: Router,
        task_name: str,
        mode: Literal["ask"],
        response_model: type[R] = TextResult,
    ) -> str: ...

    @overload
    async def __call__[R: Result](
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        model: LLMModelAlias,
        router: Router,
        task_name: str,
        response_model: type[R],
        mode: Literal["refine"],
    ) -> R: ...

    @overload
    async def __call__[R: Result](
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        model: LLMModelAlias,
        router: Router,
        task_name: str,
        response_model: type[R],
    ) -> R: ...

    async def __call__[R: Result](
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        model: LLMModelAlias,
        router: Router,
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
        dumped_result = result.model_dump_json()
        for replacement in find_replace_response.replacements:
            dumped_result = dumped_result.replace(replacement.find, replacement.replace)

        return response_model.model_validate_json(dumped_result)

    def __find_last_result[R: Result](
        self, *, messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage], response_model: type[R]
    ) -> R:
        for message in reversed(messages):
            if isinstance(message, LLMAssistantMessage):
                try:
                    return response_model.model_validate_json(message.parts[0].content)
                except ValidationError:
                    continue

        raise Honk("No last result found, failed to refine")
