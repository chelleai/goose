from typing import Literal, NewType

from pydantic import BaseModel, Field, computed_field, field_validator

JsonPath = NewType("JsonPath", str)
SourceType = Literal["task", "input"]
InputType = Literal["text", "media"]


class TextArgument(str):
    pass


class FlowArgumentSpec(BaseModel):
    parameter: str
    source_type: SourceType
    source: str
    path: JsonPath | None = None
    mapped: bool = False


class FlowConditionSpec(BaseModel):
    parameter: str
    match_path: str
    match: str


class FlowTaskSpec(BaseModel):
    name: str
    arguments: list[FlowArgumentSpec]
    conditions: list[FlowConditionSpec] | None = Field(default=None, alias="if")

    @field_validator("arguments", mode="after")
    @classmethod
    def at_most_one_mapped_argument(cls, arguments: list[FlowArgumentSpec]) -> list[FlowArgumentSpec]:
        mapped_arguments = [argument for argument in arguments if argument.mapped]
        if len(mapped_arguments) > 1:
            raise ValueError("At most one mapped argument is allowed")
        return arguments

    @computed_field
    @property
    def inbound_task_names(self) -> list[str]:
        return [argument.source for argument in self.arguments if argument.source_type == "task"]


class FlowInputSpec(BaseModel):
    name: str
    type: InputType
    description: str


class FlowSpec(BaseModel):
    name: str
    inputs: list[FlowInputSpec]
    tasks: list[FlowTaskSpec]

    @field_validator("tasks", mode="after")
    @classmethod
    def unique_task_names(cls, tasks: list[FlowTaskSpec]) -> list[FlowTaskSpec]:
        task_names = {task.name for task in tasks}
        if len(task_names) != len(tasks):
            raise ValueError("Task names must be unique")

        return tasks

    @field_validator("inputs", mode="after")
    @classmethod
    def unique_input_names(cls, inputs: list[FlowInputSpec]) -> list[FlowInputSpec]:
        input_names = {input.name for input in inputs}
        if len(input_names) != len(inputs):
            raise ValueError("Input names must be unique")

        return inputs


class FlowsSpec(BaseModel):
    flows: list[FlowSpec]
