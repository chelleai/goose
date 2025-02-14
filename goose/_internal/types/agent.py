import base64
from enum import StrEnum
from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel


class AIModel(StrEnum):
    # vertex (production Google, requires GCP environment)
    VERTEX_PRO = "vertex_ai/gemini-1.5-pro"
    VERTEX_FLASH = "vertex_ai/gemini-1.5-flash"
    VERTEX_FLASH_8B = "vertex_ai/gemini-1.5-flash-8b"
    VERTEX_FLASH_2_0 = "vertex_ai/gemini-2.0-flash"

    # gemini (publicly available, no GCP environment required)
    GEMINI_PRO = "gemini/gemini-1.5-pro"
    GEMINI_FLASH = "gemini/gemini-1.5-flash"
    GEMINI_FLASH_8B = "gemini/gemini-1.5-flash-8b"
    GEMINI_FLASH_2_0 = "gemini/gemini-2.0-flash"


class ContentType(StrEnum):
    # text
    TEXT = "text/plain"

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


class MessagePart(BaseModel):
    content: str
    content_type: ContentType = ContentType.TEXT

    @classmethod
    def from_media(cls, *, content: bytes, content_type: ContentType) -> "MessagePart":
        return cls(content=base64.b64encode(content).decode(), content_type=content_type)

    def render(self) -> LLMTextMessagePart | LLMMediaMessagePart:
        if self.content_type == ContentType.TEXT:
            return {"type": "text", "text": self.content}
        else:
            return {"type": "image_url", "image_url": f"data:{self.content_type};base64,{self.content}"}


class UserMessage(BaseModel):
    parts: list[MessagePart]

    def render(self) -> LLMMessage:
        content: LLMMessage = {
            "role": "user",
            "content": [part.render() for part in self.parts],
        }
        if any(part.content_type != ContentType.TEXT for part in self.parts):
            content["cache_control"] = {"type": "ephemeral"}
        return content


class AssistantMessage(BaseModel):
    text: str

    def render(self) -> LLMMessage:
        return {"role": "assistant", "content": [{"type": "text", "text": self.text}]}


class SystemMessage(BaseModel):
    parts: list[MessagePart]

    def render(self) -> LLMMessage:
        return {
            "role": "system",
            "content": [part.render() for part in self.parts],
        }
