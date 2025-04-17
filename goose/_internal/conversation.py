from typing import Self

from pydantic import BaseModel

from aikernel import LLMAssistantMessage, LLMSystemMessage, LLMUserMessage
from goose.errors import GooseError


class Conversation(BaseModel):
    user_messages: list[LLMUserMessage]
    assistant_messages: list[LLMAssistantMessage]
    context: LLMSystemMessage | None = None

    @property
    def awaiting_response(self) -> bool:
        return len(self.user_messages) == len(self.assistant_messages)

    def render(self) -> list[LLMSystemMessage | LLMUserMessage | LLMAssistantMessage]:
        messages: list[LLMSystemMessage | LLMUserMessage | LLMAssistantMessage] = []
        if self.context is not None:
            messages.append(self.context)

        for message_index in range(len(self.user_messages)):
            message = self.assistant_messages[message_index]
            messages.append(message)
            messages.append(self.user_messages[message_index])

        if len(self.assistant_messages) > len(self.user_messages):
            message = self.assistant_messages[-1]
            messages.append(message)

        return messages

    def undo(self) -> Self:
        if len(self.user_messages) == 0:
            raise GooseError("Cannot undo, no user messages")

        if len(self.assistant_messages) == 0:
            raise GooseError("Cannot undo, no assistant messages")

        self.user_messages.pop()
        self.assistant_messages.pop()
        return self
