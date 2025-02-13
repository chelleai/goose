from typing import Any, Literal, NotRequired, TypedDict

_LiteLLMGeminiModel = Literal[
    "vertex_ai/gemini-1.5-flash",
    "vertex_ai/gemini-1.5-pro",
    "vertex_ai/gemini-1.5-flash-8b",
    "vertex_ai/gemini-2.0-flash",
    "gemini/gemini-1.5-flash",
    "gemini/gemini-1.5-pro",
    "gemini/gemini-1.5-flash-8b",
    "gemini/gemini-2.0-flash",
]
_MessageRole = Literal["system", "user", "assistant"]

class _LiteLLMTextMessageContent(TypedDict):
    type: Literal["text"]
    text: str

class _LiteLLMMediaMessageContent(TypedDict):
    type: Literal["image_url"]
    image_url: str

class _LiteLLMCacheControl(TypedDict):
    type: Literal["ephemeral"]

class _LiteLLMMessage(TypedDict):
    role: _MessageRole
    content: list[_LiteLLMTextMessageContent | _LiteLLMMediaMessageContent]
    cache_control: NotRequired[_LiteLLMCacheControl]

class _LiteLLMResponseFormat(TypedDict):
    type: Literal["json_object"]
    response_schema: dict[str, Any]  # must be a valid JSON schema
    enforce_validation: NotRequired[bool]

class _LiteLLMModelResponseChoiceMessage:
    role: Literal["assistant"]
    content: str

class _LiteLLMModelResponseChoice:
    finish_reason: Literal["stop"]
    index: int
    message: _LiteLLMModelResponseChoiceMessage

class _LiteLLMUsage:
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int

class ModelResponse:
    id: str
    created: int
    model: _LiteLLMGeminiModel
    object: Literal["chat.completion"]
    system_fingerprint: str | None
    choices: list[_LiteLLMModelResponseChoice]
    usage: _LiteLLMUsage

async def acompletion(
    *,
    model: _LiteLLMGeminiModel,
    messages: list[_LiteLLMMessage],
    response_format: _LiteLLMResponseFormat | None = None,
    max_tokens: int | None = None,
    temperature: float = 1.0,
) -> ModelResponse: ...
