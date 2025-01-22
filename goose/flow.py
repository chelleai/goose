from enum import StrEnum

from pydantic import BaseModel


class Flow[ParametersT: BaseModel, TasksT: StrEnum]:
    def __init__(
        self, *, parameter_model: type[ParametersT], tasks: type[TasksT]
    ) -> None:
        self.parameter_model = parameter_model
        self.tasks = tasks

    def compile(
        self, arguments: ParametersT, /, *, unlocked: list[TasksT] | None = None
    ) -> None:
        pass

    async def run(self) -> None:
        pass
