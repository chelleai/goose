import json
from datetime import datetime
from typing import ClassVar, TypedDict

from pydantic import BaseModel, computed_field

from ..types.agent import AIModel, LLMMessage


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
    INPUT_DOLLARS_PER_MILLION_TOKENS: ClassVar[dict[AIModel, float]] = {
        AIModel.VERTEX_FLASH_8B: 0.30,
        AIModel.VERTEX_FLASH: 0.15,
        AIModel.VERTEX_PRO: 5.00,
        AIModel.GEMINI_FLASH_8B: 0.30,
        AIModel.GEMINI_FLASH: 0.15,
        AIModel.GEMINI_PRO: 5.00,
    }
    OUTPUT_DOLLARS_PER_MILLION_TOKENS: ClassVar[dict[AIModel, float]] = {
        AIModel.VERTEX_FLASH_8B: 0.30,
        AIModel.VERTEX_FLASH: 0.15,
        AIModel.VERTEX_PRO: 5.00,
        AIModel.GEMINI_FLASH_8B: 0.30,
        AIModel.GEMINI_FLASH: 0.15,
        AIModel.GEMINI_PRO: 5.00,
    }

    response: R
    run_id: str
    flow_name: str
    task_name: str
    model: AIModel
    system: LLMMessage | None = None
    input_messages: list[LLMMessage]
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
        return self.INPUT_DOLLARS_PER_MILLION_TOKENS[self.model] * self.input_tokens / 1_000_000

    @computed_field
    @property
    def output_cost(self) -> float:
        return self.OUTPUT_DOLLARS_PER_MILLION_TOKENS[self.model] * self.output_tokens / 1_000_000

    @computed_field
    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost

    def minimized_dump(self) -> AgentResponseDump:
        if self.system is None:
            minimized_system_message = ""
        else:
            minimized_system_message = self.system
            for part in minimized_system_message["content"]:
                if part["type"] == "image_url":
                    part["image_url"] = "__MEDIA__"
            minimized_system_message = json.dumps(minimized_system_message)

        minimized_input_messages = [message for message in self.input_messages]
        for message in minimized_input_messages:
            for part in message["content"]:
                if part["type"] == "image_url":
                    part["image_url"] = "__MEDIA__"
        minimized_input_messages = [json.dumps(message) for message in minimized_input_messages]

        output_message = self.response.model_dump_json() if isinstance(self.response, BaseModel) else self.response

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
