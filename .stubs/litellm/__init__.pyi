from typing import Any, Literal, NotRequired, TypedDict

modify_params: bool

_MessageRole = Literal["system", "user", "assistant", "tool"]


# request
class _LiteLLMTextMessageContent(TypedDict):
    type: Literal["text"]
    text: str


class _LiteLLMMediaMessageContent(TypedDict):
    type: Literal["image_url"]
    image_url: str


class _LiteLLMToolMessageContent(TypedDict):
    type: Literal["tool"]
    tool_call_id: str
    name: str
    content: str


class _LiteLLMCacheControl(TypedDict):
    type: Literal["ephemeral"]


class _LiteLLMFunctionCall(TypedDict):
    name: str
    arguments: str


class _LiteLLMToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: _LiteLLMFunctionCall


class _LiteLLMMessage(TypedDict):
    role: _MessageRole
    tool_call_id: NotRequired[str]
    name: NotRequired[str]
    content: list[_LiteLLMTextMessageContent | _LiteLLMMediaMessageContent] | str | None
    tool_calls: NotRequired[list[_LiteLLMToolCall]]
    cache_control: NotRequired[_LiteLLMCacheControl]


class _LiteLLMFunction(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]


class _LiteLLMTool(TypedDict):
    type: Literal["function"]
    function: _LiteLLMFunction


# response
class _LiteLLMModelResponseChoiceToolCallFunction:
    name: str
    arguments: str


class _LiteLLMModelResponseChoiceToolCall:
    id: str
    function: _LiteLLMModelResponseChoiceToolCallFunction
    type: Literal["function"]


class _LiteLLMModelResponseChoiceMessage:
    role: Literal["assistant"]
    content: str
    tool_calls: list[_LiteLLMModelResponseChoiceToolCall] | None

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
    model: str
    object: Literal["chat.completion"]
    system_fingerprint: str | None
    choices: list[_LiteLLMModelResponseChoice]
    usage: _LiteLLMUsage


class _LiteLLMEmbeddingData(TypedDict):
    index: int
    object: Literal["embedding"]
    embedding: list[float]


class EmbeddingResponse:
    data: list[_LiteLLMEmbeddingData]
    model: str
    usage: _LiteLLMUsage


def completion(
    *,
    model: str,
    messages: list[_LiteLLMMessage],
    response_format: Any = None,
    tools: list[_LiteLLMTool] | None = None,
    tool_choice: Literal["auto", "required"] | None = None,
    max_tokens: int | None = None,
    temperature: float = 1.0,
    num_retries: int = 0,
) -> ModelResponse: ...


async def acompletion(
    *,
    model: str,
    messages: list[_LiteLLMMessage],
    response_format: Any = None,
    tools: list[_LiteLLMTool] | None = None,
    tool_choice: Literal["auto", "required"] | None = None,
    max_tokens: int | None = None,
    temperature: float = 1.0,
    num_retries: int = 0,
) -> ModelResponse: ...

def embedding(model: str, input: list[str]) -> EmbeddingResponse: ...

async def aembedding(model: str, input: list[str]) -> EmbeddingResponse: ...


# router
class _LiteLLMRouterModelParams(TypedDict):
    model: str
    api_base: NotRequired[str]
    api_key: NotRequired[str]
    rpm: NotRequired[int]


class _LiteLLMRouterModel(TypedDict):
    model_name: str
    litellm_params: _LiteLLMRouterModelParams


class Router:
    model_names: list[str]

    def __init__(self, *, model_list: list[_LiteLLMRouterModel], fallbacks: list[dict[str, list[str]]]) -> None: ...

    async def acompletion(
        self,
        *,
        model: str,
        messages: list[_LiteLLMMessage],
        response_format: Any = None,
        tools: list[_LiteLLMTool] | None = None,
        tool_choice: Literal["auto", "required"] | None = None,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        num_retries: int = 0,
    ) -> ModelResponse: ...

    def completion(
        self,
        *,
        model: str,
        messages: list[_LiteLLMMessage],
        response_format: Any = None,
        tools: list[_LiteLLMTool] | None = None,
        tool_choice: Literal["auto", "required"] | None = None,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        num_retries: int = 0,
    ) -> ModelResponse: ...