from typing import Self

from pydantic import BaseModel

from goose.errors import Honk

from .result import Result
from .types.agent import AssistantMessage, LLMMessage, SystemMessage, UserMessage


class Conversation[R: Result](BaseModel):
    user_messages: list[UserMessage]
    assistant_messages: list[R | str]
    context: SystemMessage | None = None

    @property
    def awaiting_response(self) -> bool:
        return len(self.user_messages) == len(self.assistant_messages)

    def render(self) -> list[LLMMessage]:
        messages: list[LLMMessage] = []
        if self.context is not None:
            messages.append(self.context.render())

        for message_index in range(len(self.user_messages)):
            message = self.assistant_messages[message_index]
            if isinstance(message, str):
                messages.append(AssistantMessage(text=message).render())
            else:
                messages.append(AssistantMessage(text=message.model_dump_json()).render())

            messages.append(self.user_messages[message_index].render())

        if len(self.assistant_messages) > len(self.user_messages):
            message = self.assistant_messages[-1]
            if isinstance(message, str):
                messages.append(AssistantMessage(text=message).render())
            else:
                messages.append(AssistantMessage(text=message.model_dump_json()).render())

        return messages

    def undo(self) -> Self:
        if len(self.user_messages) == 0:
            raise Honk("Cannot undo, no user messages")

        if len(self.assistant_messages) == 0:
            raise Honk("Cannot undo, no assistant messages")

        self.user_messages.pop()
        self.assistant_messages.pop()
        return self
