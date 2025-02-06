import base64
from enum import StrEnum
from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel


class Base64MediaContent(str):
    @classmethod
    def from_bytes(cls, content: bytes, /) -> "Base64MediaContent":
        return cls(base64.b64encode(content).decode())


class AIModel(StrEnum):
    # vertex (production Google, requires GCP environment)
    VERTEX_PRO = "vertex_ai/gemini-1.5-pro"
    VERTEX_FLASH = "vertex_ai/gemini-1.5-flash"
    VERTEX_FLASH_8B = "vertex_ai/gemini-1.5-flash-8b"

    # gemini (publicly available, no GCP environment required)
    GEMINI_PRO = "gemini/gemini-1.5-pro"
    GEMINI_FLASH = "gemini/gemini-1.5-flash"
    GEMINI_FLASH_8B = "gemini/gemini-1.5-flash-8b"


class UserMediaContentType(StrEnum):
    # images
    JPEG = "image/jpeg"
    PNG = "image/png"
    WEBP = "image/webp"

    # audio
    MP3 = "audio/mp3"
    WAV = "audio/wav"

    # files
    PDF = "application/pdf"


class LLMTextMessagePart(TypedDict):
    type: Literal["text"]
    text: str


class LLMMediaMessagePart(TypedDict):
    type: Literal["image_url"]
    image_url: str


class CacheControl(TypedDict):
    type: Literal["ephemeral"]


class LLMMessage(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: list[LLMTextMessagePart | LLMMediaMessagePart]
    cache_control: NotRequired[CacheControl]


class TextMessagePart(BaseModel):
    text: str

    def render(self) -> LLMTextMessagePart:
        return {"type": "text", "text": self.text}


class MediaMessagePart(BaseModel):
    content_type: UserMediaContentType
    content: Base64MediaContent

    def render(self) -> LLMMediaMessagePart:
        return {
            "type": "image_url",
            "image_url": f"data:{self.content_type};base64,{self.content}",
        }


class UserMessage(BaseModel):
    parts: list[TextMessagePart | MediaMessagePart]

    def render(self) -> LLMMessage:
        content: LLMMessage = {
            "role": "user",
            "content": [part.render() for part in self.parts],
        }
        if any(isinstance(part, MediaMessagePart) for part in self.parts):
            content["cache_control"] = {"type": "ephemeral"}
        return content


class AssistantMessage(BaseModel):
    text: str

    def render(self) -> LLMMessage:
        return {"role": "assistant", "content": [{"type": "text", "text": self.text}]}


class SystemMessage(BaseModel):
    parts: list[TextMessagePart | MediaMessagePart]

    def render(self) -> LLMMessage:
        return {
            "role": "system",
            "content": [part.render() for part in self.parts],
        }
