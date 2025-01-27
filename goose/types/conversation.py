from pydantic import BaseModel, ConfigDict, field_validator

from goose.errors import Honk
from goose.types.agent import UserMessage


class Result(BaseModel):
    model_config = ConfigDict(frozen=True)


class GooseResponse[R: Result](BaseModel):
    result: R


class ConversationState[R: Result](BaseModel):
    messages: list[UserMessage | GooseResponse[R]]

    @field_validator("messages")
    def alternates_starting_with_result(
        cls, messages: list[UserMessage | GooseResponse[R]]
    ) -> list[UserMessage | GooseResponse[R]]:
        if len(messages) == 0:
            return messages
        elif isinstance(messages[0], UserMessage):
            raise Honk(
                "User cannot start a conversation on a Task, must begin with a Result"
            )

        last_message_type: type[UserMessage | GooseResponse[R]] = type(messages[0])
        for message in messages:
            if isinstance(message, last_message_type):
                raise Honk(
                    "Conversation must alternate between User and Result messages"
                )
            last_message_type = type(message)

        return messages

    @property
    def awaiting_response(self) -> bool:
        return len(self.messages) % 2 == 0
