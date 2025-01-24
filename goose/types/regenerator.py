from pydantic import BaseModel

from goose.types.messages import UserMessage


class ConversationRound[R](BaseModel):
    user_message: UserMessage
    result: R


class Conversation[R](BaseModel):
    initial_result: R
    rounds: list[ConversationRound[R]]

    @property
    def current_result(self) -> R:
        if len(self.rounds) == 0:
            return self.initial_result
        return self.rounds[-1].result
