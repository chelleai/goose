from typing import Literal

from pydantic import BaseModel, ConfigDict

from vers.types.llm import GeminiModel, MessageRole


class Parameter(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    type: Literal["text", "media"]
    description: str


class TaskMessage(BaseModel):
    role: MessageRole
    content: str
    cached: bool = False


class TaskSpec(BaseModel):
    name: str
    model: GeminiModel
    response_model_name: str | None = None
    parameters: list[Parameter] | None = None
    messages: list[TaskMessage]


class TasksSpec(BaseModel):
    tasks: list[TaskSpec]


class TaskOutput(BaseModel):
    pass


class UnstructuredTaskOutput(TaskOutput):
    content: str
