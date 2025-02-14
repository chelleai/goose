import json
import logging
from datetime import datetime
from typing import Any, ClassVar, Protocol, TypedDict

from litellm import acompletion
from pydantic import BaseModel, computed_field

from .result import Result, TextResult
from .types.agent import AIModel, AssistantMessage, SystemMessage, UserMessage


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
    INPUT_CENTS_PER_MILLION_TOKENS: ClassVar[dict[AIModel, float]] = {
        AIModel.VERTEX_FLASH_8B: 30,
        AIModel.VERTEX_FLASH: 15,
        AIModel.VERTEX_PRO: 500,
        AIModel.GEMINI_FLASH_8B: 30,
        AIModel.GEMINI_FLASH: 15,
        AIModel.GEMINI_PRO: 500,
    }
    OUTPUT_CENTS_PER_MILLION_TOKENS: ClassVar[dict[AIModel, float]] = {
        AIModel.VERTEX_FLASH_8B: 30,
        AIModel.VERTEX_FLASH: 15,
        AIModel.VERTEX_PRO: 500,
        AIModel.GEMINI_FLASH_8B: 30,
        AIModel.GEMINI_FLASH: 15,
        AIModel.GEMINI_PRO: 500,
    }

    response: R
    run_id: str
    flow_name: str
    task_name: str
    model: AIModel
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
        return self.INPUT_CENTS_PER_MILLION_TOKENS[self.model] * self.input_tokens / 1_000_000

    @computed_field
    @property
    def output_cost(self) -> float:
        return self.OUTPUT_CENTS_PER_MILLION_TOKENS[self.model] * self.output_tokens / 1_000_000

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
        model: AIModel,
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
            parsed_response = response_model.model_validate({"text": response.choices[0].message.content})
        else:
            response = await acompletion(
                model=model.value,
                messages=rendered_messages,
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
