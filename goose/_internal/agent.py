import logging
from datetime import datetime
from typing import Any, Literal, Protocol, overload

from litellm import acompletion
from pydantic import ValidationError

from goose._internal.types.telemetry import AgentResponse
from goose.errors import Honk

from .result import FindReplaceResponse, Result, TextResult
from .types.agent import AIModel, LLMMessage


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
        messages: list[LLMMessage],
        model: AIModel,
        task_name: str,
        response_model: type[R] = TextResult,
        system: LLMMessage | None = None,
    ) -> R:
        start_time = datetime.now()
        if system is not None:
            messages.insert(0, system)

        if response_model is TextResult:
            response = await acompletion(model=model.value, messages=messages)
            parsed_response = response_model.model_validate({"text": response.choices[0].message.content})
        else:
            response = await acompletion(
                model=model.value,
                messages=messages,
                response_format=response_model,
            )
            parsed_response = response_model.model_validate_json(response.choices[0].message.content)

        end_time = datetime.now()
        agent_response = AgentResponse(
            response=parsed_response,
            run_id=self.run_id,
            flow_name=self.flow_name,
            task_name=task_name,
            model=model,
            system=system,
            input_messages=messages,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            start_time=start_time,
            end_time=end_time,
        )

        if self.logger is not None:
            await self.logger(response=agent_response)
        else:
            logging.info(agent_response.model_dump())

        return parsed_response

    async def ask(
        self, *, messages: list[LLMMessage], model: AIModel, task_name: str, system: LLMMessage | None = None
    ) -> str:
        start_time = datetime.now()

        if system is not None:
            messages.insert(0, system)
        response = await acompletion(model=model.value, messages=messages)

        end_time = datetime.now()
        agent_response = AgentResponse(
            response=response.choices[0].message.content,
            run_id=self.run_id,
            flow_name=self.flow_name,
            task_name=task_name,
            model=model,
            system=system,
            input_messages=messages,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            start_time=start_time,
            end_time=end_time,
        )

        if self.logger is not None:
            await self.logger(response=agent_response)
        else:
            logging.info(agent_response.model_dump())

        return response.choices[0].message.content

    async def refine[R: Result](
        self,
        *,
        messages: list[LLMMessage],
        model: AIModel,
        task_name: str,
        response_model: type[R],
        system: LLMMessage | None = None,
    ) -> R:
        start_time = datetime.now()

        if system is not None:
            messages.insert(0, system)

        find_replace_response = await acompletion(
            model=model.value, messages=messages, response_format=FindReplaceResponse
        )
        parsed_find_replace_response = FindReplaceResponse.model_validate_json(
            find_replace_response.choices[0].message.content
        )

        end_time = datetime.now()
        agent_response = AgentResponse(
            response=parsed_find_replace_response,
            run_id=self.run_id,
            flow_name=self.flow_name,
            task_name=task_name,
            model=model,
            system=system,
            input_messages=messages,
            input_tokens=find_replace_response.usage.prompt_tokens,
            output_tokens=find_replace_response.usage.completion_tokens,
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
        messages: list[LLMMessage],
        model: AIModel,
        task_name: str,
        mode: Literal["generate"],
        response_model: type[R],
        system: LLMMessage | None = None,
    ) -> R: ...

    @overload
    async def __call__[R: Result](
        self,
        *,
        messages: list[LLMMessage],
        model: AIModel,
        task_name: str,
        mode: Literal["ask"],
        response_model: type[R] = TextResult,
        system: LLMMessage | None = None,
    ) -> str: ...

    @overload
    async def __call__[R: Result](
        self,
        *,
        messages: list[LLMMessage],
        model: AIModel,
        task_name: str,
        response_model: type[R],
        mode: Literal["refine"],
        system: LLMMessage | None = None,
    ) -> R: ...

    @overload
    async def __call__[R: Result](
        self,
        *,
        messages: list[LLMMessage],
        model: AIModel,
        task_name: str,
        response_model: type[R],
        system: LLMMessage | None = None,
    ) -> R: ...

    async def __call__[R: Result](
        self,
        *,
        messages: list[LLMMessage],
        model: AIModel,
        task_name: str,
        response_model: type[R] = TextResult,
        mode: Literal["generate", "ask", "refine"] = "generate",
        system: LLMMessage | None = None,
    ) -> R | str:
        match mode:
            case "generate":
                return await self.generate(
                    messages=messages, model=model, task_name=task_name, response_model=response_model, system=system
                )
            case "ask":
                return await self.ask(messages=messages, model=model, task_name=task_name, system=system)
            case "refine":
                return await self.refine(
                    messages=messages, model=model, task_name=task_name, response_model=response_model, system=system
                )

    def __apply_find_replace[R: Result](
        self, *, result: R, find_replace_response: FindReplaceResponse, response_model: type[R]
    ) -> R:
        dumped_result = result.model_dump_json()
        for replacement in find_replace_response.replacements:
            dumped_result = dumped_result.replace(replacement.find, replacement.replace)

        return response_model.model_validate_json(dumped_result)

    def __find_last_result[R: Result](self, *, messages: list[LLMMessage], response_model: type[R]) -> R:
        for message in reversed(messages):
            if message["role"] == "assistant":
                try:
                    only_part = message["content"][0]
                    if only_part["type"] == "text":
                        return response_model.model_validate_json(only_part["text"])
                except ValidationError:
                    continue

        raise Honk("No last result found, failed to refine")
