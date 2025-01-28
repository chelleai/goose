import base64
import logging
from datetime import datetime
from enum import StrEnum
from typing import Any, Awaitable, Callable, ClassVar, Literal, NotRequired, TypedDict

from litellm import acompletion
from pydantic import BaseModel, computed_field


class GeminiModel(StrEnum):
    EXP = "gemini/gemini-exp-1121"
    PRO = "gemini/gemini-1.5-pro"
    FLASH = "gemini/gemini-1.5-flash"
    FLASH_8B = "gemini/gemini-1.5-flash-8b"


class UserMediaContentType(StrEnum):
    # images
    JPEG = "image/jpeg"
    PNG = "image/png"
    WEBP = "image/webp"

    # audio
    MP3 = "audio/mpeg"
    WAV = "audio/wav"

    # files
    PDF = "application/pdf"


class LLMTextMessagePart(TypedDict):
    type: Literal["text"]
    text: str


class LLMMediaMessagePart(TypedDict):
    type: Literal["image_url"]
    image_url: str


class CacheControl(TypedDict):
    type: Literal["ephemeral"]


class LLMMessage(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: list[LLMTextMessagePart | LLMMediaMessagePart]
    cache_control: NotRequired[CacheControl]


class TextMessagePart(BaseModel):
    text: str

    def render(self) -> LLMTextMessagePart:
        return {"type": "text", "text": self.text}


class MediaMessagePart(BaseModel):
    content_type: UserMediaContentType
    content: bytes

    def render(self) -> LLMMediaMessagePart:
        return {
            "type": "image_url",
            "image_url": f"data:{self.content_type};base64,{base64.b64encode(self.content).decode()}",
        }


class UserMessage(BaseModel):
    parts: list[TextMessagePart | MediaMessagePart]

    def render(self) -> LLMMessage:
        content: LLMMessage = {
            "role": "user",
            "content": [part.render() for part in self.parts],
        }
        if any(isinstance(part, MediaMessagePart) for part in self.parts):
            content["cache_control"] = {"type": "ephemeral"}
        return content


class AssistantMessage(BaseModel):
    text: str

    def render(self) -> LLMMessage:
        return {"role": "assistant", "content": [{"type": "text", "text": self.text}]}


class SystemMessage(BaseModel):
    parts: list[TextMessagePart | MediaMessagePart]

    def render(self) -> LLMMessage:
        return {
            "role": "system",
            "content": [part.render() for part in self.parts],
        }


class AgentResponse[R: BaseModel](BaseModel):
    INPUT_CENTS_PER_MILLION_TOKENS: ClassVar[dict[GeminiModel, float]] = {
        GeminiModel.FLASH_8B: 30,
        GeminiModel.FLASH: 15,
        GeminiModel.PRO: 500,
        GeminiModel.EXP: 0,
    }
    OUTPUT_CENTS_PER_MILLION_TOKENS: ClassVar[dict[GeminiModel, float]] = {
        GeminiModel.FLASH_8B: 30,
        GeminiModel.FLASH: 15,
        GeminiModel.PRO: 500,
        GeminiModel.EXP: 0,
    }

    response: R
    run_name: str
    flow_name: str
    task_name: str
    model: GeminiModel
    system: SystemMessage | None = None
    input_messages: list[UserMessage | AssistantMessage]
    input_tokens: int
    output_tokens: int
    start_time: datetime
    end_time: datetime

    @computed_field
    @property
    def duration_ms(self) -> int:
        return int((self.end_time - self.start_time).total_seconds() * 1000)

    @computed_field
    @property
    def input_cost(self) -> float:
        return self.INPUT_CENTS_PER_MILLION_TOKENS[self.model] * self.input_tokens

    @computed_field
    @property
    def output_cost(self) -> float:
        return self.OUTPUT_CENTS_PER_MILLION_TOKENS[self.model] * self.output_tokens

    @computed_field
    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost


class Agent:
    def __init__(
        self,
        *,
        flow_name: str,
        run_name: str,
        logger: Callable[[AgentResponse[Any]], Awaitable[None]] | None = None,
    ) -> None:
        self.flow_name = flow_name
        self.run_name = run_name
        self.logger = logger

    async def __call__[R: BaseModel](
        self,
        *,
        messages: list[UserMessage | AssistantMessage],
        model: GeminiModel,
        response_model: type[R],
        task_name: str,
        system: SystemMessage | None = None,
    ) -> R:
        start_time = datetime.now()
        rendered_messages = [message.render() for message in messages]
        if system is not None:
            rendered_messages.insert(0, system.render())

        response = await acompletion(
            model=model.value,
            messages=rendered_messages,
            response_format={
                "type": "json_object",
                "response_schema": response_model.model_json_schema(),
                "enforce_validation": True,
            },
        )

        if len(response.choices) == 0:
            raise RuntimeError("No content returned from LLM call.")

        parsed_response = response_model.model_validate_json(
            response.choices[0].message.content
        )
        end_time = datetime.now()
        agent_response = AgentResponse(
            response=parsed_response,
            run_name=self.run_name,
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
            await self.logger(agent_response)
        else:
            logging.info(agent_response.model_dump())

        return agent_response.response
