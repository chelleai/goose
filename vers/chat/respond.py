from litellm import acompletion

from vers.chat.system import SystemPrompt
from vers.errors import VersError
from vers.types.llm import (
    GeminiModel,
    LLMMessage,
)


async def chat_response(
    *, system_prompt: SystemPrompt, messages: list[LLMMessage], model: GeminiModel
) -> str:
    all_messages = [system_prompt.render(), *messages]
    response = await acompletion(model=model.value, messages=all_messages)
    if len(response.choices) == 0:
        raise VersError(
            error_type="LLM_ERROR", message="No content returned from LLM call."
        )

    return response.choices[0].message.content
