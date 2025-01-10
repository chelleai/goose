import json
import logging
import os
from datetime import datetime

from litellm import acompletion
from pydantic import BaseModel, computed_field

from vers.types.llm import GeminiModel, LLMMessage
from vers.types.tasks import UnstructuredTaskOutput


class TaskLog(BaseModel):
    run_id: str
    map_index: str | None
    flow_name: str
    task_name: str
    model: GeminiModel
    input_messages: str
    output_message: str
    input_cost: float
    output_cost: float
    start_time: datetime
    end_time: datetime

    @computed_field
    @property
    def service(self) -> str:
        return os.getenv("SERVICE", "")

    @computed_field
    @property
    def duration_ms(self) -> int:
        return int((self.end_time - self.start_time).total_seconds() * 1000)

    @computed_field
    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost


class TaskRunner:
    TRACING_DATASET_ID = "tracing"
    TRACING_TABLE_ID = "llm_tasks"

    def __init__(
        self,
        *,
        flow_name: str,
        run_id: str,
        map_index: str | None,
        disable_tracing: bool,
    ) -> None:
        self.flow_name = flow_name
        self.run_id = run_id
        self.map_index = map_index
        self.disable_tracing = disable_tracing
        self.logger = logging.getLogger("vers.runner")

    async def create(
        self, *, messages: list[LLMMessage], model: GeminiModel, task_name: str
    ) -> UnstructuredTaskOutput:
        start_time = datetime.now()

        response = await acompletion(model=model.value, messages=messages)
        if len(response.choices) == 0:
            raise ValueError("No content returned from LLM call.")
        response_content = response.choices[0].message.content

        if not self.disable_tracing:
            await self.__log_trace(
                task_name=task_name,
                model=model,
                input_messages=messages,
                output_message=response_content,
                input_cost=self.__get_input_cost(
                    tokens=response.usage.prompt_tokens, model=model
                ),
                output_cost=self.__get_output_cost(
                    tokens=response.usage.completion_tokens, model=model
                ),
                start_time=start_time,
                end_time=datetime.now(),
            )

        return UnstructuredTaskOutput(content=response_content)

    async def parse[ResponseModelT: BaseModel](
        self,
        *,
        messages: list[LLMMessage],
        model: GeminiModel,
        response_model: type[ResponseModelT],
        task_name: str,
    ) -> ResponseModelT:
        start_time = datetime.now()

        response = await acompletion(
            model=model.value,
            messages=messages,
            response_format={
                "type": "json_object",
                "response_schema": response_model.model_json_schema(),
                "enforce_validation": True,
            },
        )

        if len(response.choices) == 0:
            raise ValueError("No content returned from LLM call.")
        parsed_response = response_model.model_validate_json(
            response.choices[0].message.content
        )

        input_cost = self.__get_input_cost(
            tokens=response.usage.prompt_tokens, model=model
        )
        output_cost = self.__get_output_cost(
            tokens=response.usage.completion_tokens, model=model
        )

        if not self.disable_tracing:
            await self.__log_trace(
                task_name=task_name,
                model=model,
                input_messages=messages,
                output_message=parsed_response.model_dump_json(),
                input_cost=input_cost,
                output_cost=output_cost,
                start_time=start_time,
                end_time=datetime.now(),
            )
        return parsed_response

    async def __log_trace(
        self,
        *,
        task_name: str,
        model: GeminiModel,
        input_messages: list[LLMMessage],
        output_message: str,
        input_cost: float,
        output_cost: float,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        for message in input_messages:
            for content_part in message["content"]:
                if content_part["type"] == "image_url":
                    content_part["image_url"] = "__TRUNCATED_FILE_CONTENT__"

        log = TaskLog(
            run_id=self.run_id,
            map_index=self.map_index,
            flow_name=self.flow_name,
            task_name=task_name,
            model=model,
            input_messages=json.dumps(input_messages),
            output_message=output_message,
            input_cost=input_cost,
            output_cost=output_cost,
            start_time=start_time,
            end_time=end_time,
        ).model_dump()
        self.logger.info(log)

    def __get_input_cost(self, *, tokens: int, model: GeminiModel) -> float:
        cents_per_million_tokens = {
            GeminiModel.FLASH_8B: 3.75,
            GeminiModel.FLASH: 7.5,
            GeminiModel.PRO: 125,
            GeminiModel.EXP: 0,
        }
        return (cents_per_million_tokens[model] / 100) * (tokens / 1_000_000)

    def __get_output_cost(self, *, tokens: int, model: GeminiModel) -> float:
        cents_per_million_tokens = {
            GeminiModel.FLASH_8B: 30,
            GeminiModel.FLASH: 15,
            GeminiModel.PRO: 500,
            GeminiModel.EXP: 0,
        }
        return (cents_per_million_tokens[model] / 100) * (tokens / 1_000_000)
