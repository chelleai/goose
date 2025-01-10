import base64

from litellm import acompletion

from vers.chat.system import SystemPrompt
from vers.types.llm import (
    GeminiModel,
    LLMMessage,
    MediaMessageDataContent,
    TextMessageDataContent,
)
from vers.types.media import EncodedMediaMessage, MediaMessage


async def chat_response(
    *,
    system_prompt: SystemPrompt,
    previous_messages: list[LLMMessage],
    user_message: str,
    model: GeminiModel,
    media: MediaMessage | EncodedMediaMessage | None = None,
) -> str:
    new_message = LLMMessage(
        role="user", content=[TextMessageDataContent(type="text", text=user_message)]
    )
    if media is not None:
        if isinstance(media, MediaMessage):
            media_content = MediaMessageDataContent(
                type="image_url",
                image_url=f"data:{media.mime_type};base64,{base64.b64encode(media.data).decode('utf-8')}",
            )
        else:
            media_content = MediaMessageDataContent(type="image_url", image_url=media)
        new_message["content"].append(media_content)

    messages = [
        system_prompt.render(),
        *previous_messages,
        new_message,
    ]

    response = await acompletion(model=model.value, messages=messages)
    if len(response.choices) == 0:
        raise ValueError("No content returned from LLM call.")

    return response.choices[0].message.content
