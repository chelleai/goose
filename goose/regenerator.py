from goose.conversation import Conversation


async def default_regenerator[R](*, result: R, conversation: Conversation[R]) -> R:
    return result
