import json
import logging
from datetime import datetime
from enum import StrEnum
from typing import Any, ClassVar, Literal, NotRequired, Protocol, TypedDict

from litellm import acompletion
from pydantic import BaseModel, computed_field
from goose.result import Result, TextResult


class GeminiModel(StrEnum):
    PRO = "vertex_ai/gemini-1.5-pro"
    FLASH = "vertex_ai/gemini-1.5-flash"
    FLASH_8B = "vertex_ai/gemini-1.5-flash-8b"


class UserMediaContentType(StrEnum):
    # images
    JPEG = "image/jpeg"
    PNG = "image/png"
    WEBP = "image/webp"

    # audio
    MP3 = "audio/mp3"
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
    content: str

    def render(self) -> LLMMediaMessagePart:
        return {
            "type": "image_url",
            "image_url": f"data:{self.content_type};base64,{self.content}",
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


class AgentResponseDump(TypedDict):
    run_id: str
    flow_name: str
    task_name: str
    model: str
    system_message: str
    input_messages: list[str]
    output_message: str
    input_cost: float
    output_cost: float
    total_cost: float
    input_tokens: int
    output_tokens: int
    start_time: datetime
    end_time: datetime
    duration_ms: int


class AgentResponse[R: BaseModel | str](BaseModel):
    INPUT_CENTS_PER_MILLION_TOKENS: ClassVar[dict[GeminiModel, float]] = {
        GeminiModel.FLASH_8B: 30,
        GeminiModel.FLASH: 15,
        GeminiModel.PRO: 500,
    }
    OUTPUT_CENTS_PER_MILLION_TOKENS: ClassVar[dict[GeminiModel, float]] = {
        GeminiModel.FLASH_8B: 30,
        GeminiModel.FLASH: 15,
        GeminiModel.PRO: 500,
    }

    response: R
    run_id: str
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
        return (
            self.INPUT_CENTS_PER_MILLION_TOKENS[self.model]
            * self.input_tokens
            / 1_000_000
        )

    @computed_field
    @property
    def output_cost(self) -> float:
        return (
            self.OUTPUT_CENTS_PER_MILLION_TOKENS[self.model]
            * self.output_tokens
            / 1_000_000
        )

    @computed_field
    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost

    def minimized_dump(self) -> AgentResponseDump:
        if self.system is None:
            minimized_system_message = ""
        else:
            minimized_system_message = self.system.render()
            for part in minimized_system_message["content"]:
                if part["type"] == "image_url":
                    part["image_url"] = "__MEDIA__"
            minimized_system_message = json.dumps(minimized_system_message)

        minimized_input_messages = [message.render() for message in self.input_messages]
        for message in minimized_input_messages:
            for part in message["content"]:
                if part["type"] == "image_url":
                    part["image_url"] = "__MEDIA__"
        minimized_input_messages = [
            json.dumps(message) for message in minimized_input_messages
        ]

        output_message = (
            self.response.model_dump_json()
            if isinstance(self.response, BaseModel)
            else self.response
        )

        return {
            "run_id": self.run_id,
            "flow_name": self.flow_name,
            "task_name": self.task_name,
            "model": self.model.value,
            "system_message": minimized_system_message,
            "input_messages": minimized_input_messages,
            "output_message": output_message,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
        }


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

    async def __call__[R: Result](
        self,
        *,
        messages: list[UserMessage | AssistantMessage],
        model: GeminiModel,
        task_name: str,
        response_model: type[R] = TextResult,
        system: SystemMessage | None = None,
    ) -> R:
        start_time = datetime.now()
        rendered_messages = [message.render() for message in messages]
        if system is not None:
            rendered_messages.insert(0, system.render())

        if response_model is TextResult:
            response = await acompletion(model=model.value, messages=rendered_messages)
            parsed_response = response_model.model_validate(
                {"text": response.choices[0].message.content}
            )
        else:
            response = await acompletion(
                model=model.value,
                messages=rendered_messages,
                response_format={
                    "type": "json_object",
                    "response_schema": response_model.model_json_schema(),
                    "enforce_validation": True,
                },
            )
            parsed_response = response_model.model_validate_json(
                response.choices[0].message.content
            )

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
