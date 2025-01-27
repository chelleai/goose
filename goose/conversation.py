from pydantic import BaseModel

from goose.types import UserMessage


class ConversationState[R: BaseModel](BaseModel):
    user_messages: list[UserMessage]
    results: list[R]


class Conversation[R: BaseModel]:
    def __init__(
        self,
        *,
        user_messages: list[UserMessage] | None = None,
        results: list[R] | None = None,
    ) -> None:
        self.user_messages = user_messages or []
        self.results = results or []

    @classmethod
    def load(cls, *, state: ConversationState[R]) -> "Conversation[R]":
        return cls(user_messages=state.user_messages, results=state.results)

    @property
    def current_result(self) -> R:
        if len(self.results) == 0:
            raise RuntimeError("No results in conversation")

        return self.results[-1]

    @property
    def state(self) -> ConversationState[R]:
        return ConversationState(user_messages=self.user_messages, results=self.results)

    def add_message(self, *, message: UserMessage) -> None:
        self.user_messages.append(message)

    def add_result(self, *, result: R) -> None:
        self.results.append(result)

    def replace_last_result(self, *, result: R) -> None:
        if len(self.results) == 0:
            self.results.append(result)
        else:
            self.results[-1] = result
