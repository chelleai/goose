import base64
from typing import Literal

from vers.runner import TaskRunner
from vers.types.llm import (
    GeminiModel,
    LLMMessage,
    MediaMessageDataContent,
    TextMessageDataContent,
)
from vers.types.media import MediaMessage
from vers.types.tasks import (
    Parameter,
    TaskMessage,
    TaskOutput,
    UnstructuredTaskOutput,
)


class Task[ResponseModelT: TaskOutput]:
    def __init__(
        self,
        *,
        name: str,
        model: GeminiModel,
        messages: list[TaskMessage],
        parameters: list[Parameter],
        response_model: type[ResponseModelT] = UnstructuredTaskOutput,
    ) -> None:
        self.name = name
        self.model = model
        self.messages = messages
        self.parameters = parameters
        self.response_model = response_model
        self._outputs: dict[str | None, ResponseModelT] = {}
        self.skipped = False

    @property
    def outputs(self) -> dict[str | None, ResponseModelT]:
        if len(self._outputs) == 0:
            raise ValueError(f"Task {self.name} has not been run or uses mapped output")
        return self._outputs

    @property
    def output(self) -> ResponseModelT:
        return self.outputs[None]

    async def run(
        self,
        *,
        flow_name: str,
        run_id: str,
        map_index: str | None,
        disable_tracing: bool,
        **arguments: str | MediaMessage,
    ) -> None:
        runner = TaskRunner(
            flow_name=flow_name,
            run_id=run_id,
            map_index=map_index,
            disable_tracing=disable_tracing,
        )
        rendered_messages = [
            self.__render_message(message=message, **arguments)
            for message in self.messages
        ]

        if self.response_model is UnstructuredTaskOutput:
            text_output = await runner.create(
                messages=rendered_messages,
                model=self.model,
                task_name=self.name,
            )
            output = self.response_model(content=text_output.content)
        else:
            output = await runner.parse(
                messages=rendered_messages,
                model=self.model,
                response_model=self.response_model,
                task_name=self.name,
            )

        self._outputs[map_index] = output

    def __render_message(
        self,
        *,
        message: TaskMessage,
        **arguments: str | MediaMessage,
    ) -> LLMMessage:
        llm_message: LLMMessage = {"role": message.role, "content": []}
        message_content = message.content
        message_type: Literal["text", "image_url"] = "text"

        for parameter_name, argument in arguments.items():
            placeholder = f"${parameter_name}"
            if placeholder in message.content:
                if isinstance(argument, MediaMessage):
                    base64_content = base64.b64encode(argument.data).decode()
                    message_content = (
                        f"data:{argument.mime_type};base64,{base64_content}"
                    )
                    message_type = "image_url"
                else:
                    message_content = message_content.replace(
                        f"${parameter_name}", str(argument)
                    )

        if message_type == "text":
            content = TextMessageDataContent(type="text", text=message_content)
        else:
            content = MediaMessageDataContent(
                type="image_url", image_url=message_content
            )
        llm_message["content"].append(content)

        if message.cached:
            llm_message["cache_control"] = {"type": "ephemeral"}

        return llm_message
