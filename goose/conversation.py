from typing import NewType

from pydantic import BaseModel

from goose.types import SerializableResult, UserMessage

ConversationState = NewType("ConversationState", str)


class ConversationDump[R: SerializableResult](BaseModel):
    messages: list[UserMessage]
    results: list[R]


class Conversation[R: SerializableResult]:
    def __init__(
        self, *, messages: list[UserMessage] = [], results: list[R] = []
    ) -> None:
        self.messages: list[UserMessage] = messages
        self.results: list[R] = results

    @classmethod
    def load(
        cls, *, state: ConversationState, result_type: type[R]
    ) -> "Conversation[R]":
        dump = ConversationDump.model_validate_json(state)
        return cls(
            messages=dump.messages,
            results=[
                result_type.model_validate_json(result) for result in dump.results
            ],
        )

    @property
    def current_result(self) -> R:
        if len(self.results) == 0:
            raise RuntimeError("No results in conversation")

        return self.results[-1]

    def add_message(self, *, message: UserMessage) -> None:
        self.messages.append(message)

    def dump(self) -> ConversationState:
        dump = ConversationDump(messages=self.messages, results=self.results)
        return ConversationState(dump.model_dump_json())
