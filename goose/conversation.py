from typing import Any, TypedDict

from goose.types import UserMessage


class ConversationDump(TypedDict):
    messages: list[UserMessage]
    results: list[Any]


class Conversation[R]:
    def __init__(
        self, *, messages: list[UserMessage] = [], results: list[R] = []
    ) -> None:
        self.messages: list[UserMessage] = messages
        self.results: list[R] = results

    @property
    def current_result(self) -> R:
        if len(self.results) == 0:
            raise RuntimeError("No results in conversation")

        return self.results[-1]

    def add_message(self, *, message: UserMessage) -> None:
        self.messages.append(message)

    def dump(self) -> ConversationDump:
        return ConversationDump(messages=self.messages, results=self.results)
