from typing import Literal, overload

from aikernel import (
    LLMAssistantMessage,
    LLMSystemMessage,
    LLMToolMessage,
    LLMUserMessage,
    llm_structured,
    llm_unstructured,
)
from goose._internal.result import Result, TextResult
from goose._internal.types.router import AnyLLMRouter

ExpectedMessage = LLMUserMessage | LLMAssistantMessage | LLMSystemMessage | LLMToolMessage


class Agent:
    def __init__(
        self,
        *,
        flow_name: str,
        run_id: str,
    ) -> None:
        self.flow_name = flow_name
        self.run_id = run_id

    async def generate[R: Result](
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        router: AnyLLMRouter,
        response_model: type[R] = TextResult,
    ) -> R:
        typed_messages: list[ExpectedMessage] = [*messages]

        if response_model is TextResult:
            response = await llm_unstructured(messages=typed_messages, router=router)
            parsed_response = response_model.model_validate({"text": response.text})
        else:
            response = await llm_structured(messages=typed_messages, response_model=response_model, router=router)
            parsed_response = response.structured_response

        return parsed_response

    async def ask(
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        router: AnyLLMRouter,
    ) -> str:
        typed_messages: list[ExpectedMessage] = [*messages]
        response = await llm_unstructured(messages=typed_messages, router=router)
        return response.text

    async def refine[R: Result](
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        router: AnyLLMRouter,
        response_model: type[R],
    ) -> R:
        typed_messages: list[ExpectedMessage] = [*messages]
        refined_response = await llm_structured(messages=typed_messages, response_model=response_model, router=router)
        return refined_response.structured_response

    @overload
    async def __call__[R: Result](
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        router: AnyLLMRouter,
        mode: Literal["generate"],
        response_model: type[R],
    ) -> R: ...

    @overload
    async def __call__[R: Result](
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        router: AnyLLMRouter,
        mode: Literal["ask"],
        response_model: type[R] = TextResult,
    ) -> str: ...

    @overload
    async def __call__[R: Result](
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        router: AnyLLMRouter,
        response_model: type[R],
        mode: Literal["refine"],
    ) -> R: ...

    @overload
    async def __call__[R: Result](
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        router: AnyLLMRouter,
        response_model: type[R],
    ) -> R: ...

    async def __call__[R: Result](
        self,
        *,
        messages: list[LLMUserMessage | LLMAssistantMessage | LLMSystemMessage],
        router: AnyLLMRouter,
        response_model: type[R] = TextResult,
        mode: Literal["generate", "ask", "refine"] = "generate",
    ) -> R | str:
        match mode:
            case "generate":
                return await self.generate(messages=messages, router=router, response_model=response_model)
            case "ask":
                return await self.ask(messages=messages, router=router)
            case "refine":
                return await self.refine(messages=messages, router=router, response_model=response_model)
