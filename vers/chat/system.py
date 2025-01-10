import base64

from vers.types.llm import LLMMessage, MediaMessageDataContent, TextMessageDataContent
from vers.types.media import MediaMessage


class SystemPromptTemplate:
    def __init__(self, *, template: str) -> None:
        self.template = template

    def render(
        self, *, placeholders: dict[str, str | str], files: list[MediaMessage]
    ) -> LLMMessage:
        template_message = self.template
        for name, value in placeholders.items():
            placeholder = f"${name}"
            template_message = template_message.replace(placeholder, value)

        return LLMMessage(
            role="system",
            content=[
                TextMessageDataContent(type="text", text=template_message),
                *[
                    MediaMessageDataContent(
                        type="image_url",
                        image_url=f"data:{file.mime_type};base64,{base64.b64encode(file.data).decode()}",
                    )
                    for file in files
                ],
            ],
        )


class SystemPrompt:
    def __init__(
        self,
        *,
        template: SystemPromptTemplate,
        placeholders: dict[str, str],
        files: list[MediaMessage],
    ) -> None:
        self.template = template
        self.placeholders = placeholders
        self.files = files

    def render(self) -> LLMMessage:
        return self.template.render(placeholders=self.placeholders, files=self.files)
