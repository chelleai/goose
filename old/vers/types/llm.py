from enum import StrEnum
from typing import Literal, NotRequired, TypedDict

MessageRole = Literal["system", "user", "assistant"]


class GeminiModel(StrEnum):
    EXP = "gemini/gemini-exp-1121"
    PRO = "gemini/gemini-1.5-pro"
    FLASH = "gemini/gemini-1.5-flash"
    FLASH_8B = "gemini/gemini-1.5-flash-8b"


class TextMessageDataContent(TypedDict):
    type: Literal["text"]
    text: str


class MediaMessageDataContent(TypedDict):
    type: Literal["image_url"]
    image_url: str


class CacheControl(TypedDict):
    type: Literal["ephemeral"]


class LLMMessage(TypedDict):
    role: MessageRole
    content: list[TextMessageDataContent | MediaMessageDataContent]
    cache_control: NotRequired[CacheControl]
