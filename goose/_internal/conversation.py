from typing import Self

from pydantic import BaseModel

from .result import Result
from .types.agent import AssistantMessage, LLMMessage, SystemMessage, UserMessage


class Conversation[R: Result](BaseModel):
    user_messages: list[UserMessage]
    result_messages: list[R]
    context: SystemMessage | None = None

    @property
    def awaiting_response(self) -> bool:
        return len(self.user_messages) == len(self.result_messages)

    def render(self) -> list[LLMMessage]:
        messages: list[LLMMessage] = []
        if self.context is not None:
            messages.append(self.context.render())

        for message_index in range(len(self.user_messages)):
            messages.append(AssistantMessage(text=self.result_messages[message_index].model_dump_json()).render())
            messages.append(self.user_messages[message_index].render())

        if len(self.result_messages) > len(self.user_messages):
            messages.append(AssistantMessage(text=self.result_messages[-1].model_dump_json()).render())

        return messages

    def undo(self) -> Self:
        self.user_messages.pop()
        self.result_messages.pop()
        return self
