import base64
from enum import StrEnum
from typing import Literal, TypedDict

from pydantic import BaseModel


class UserMediaContentType(StrEnum):
    JPEG = "image/jpeg"
    PNG = "image/png"
    WEBP = "image/webp"
    MP3 = "audio/mpeg"
    WAV = "audio/wav"


class LLMTextMessagePart(TypedDict):
    type: Literal["text"]
    text: str


class LLMMediaMessagePart(TypedDict):
    type: Literal["image_url"]
    image_url: str


class LLMMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: list[LLMTextMessagePart | LLMMediaMessagePart]


class TextMessagePart(BaseModel):
    text: str

    def render(self) -> LLMTextMessagePart:
        return {"type": "text", "text": self.text}


class MediaMessagePart(BaseModel):
    content_type: UserMediaContentType
    content: bytes

    def render(self) -> LLMMediaMessagePart:
        return {
            "type": "image_url",
            "image_url": f"data:{self.content_type};base64,{base64.b64encode(self.content).decode()}",
        }


class UserMessage(BaseModel):
    parts: list[TextMessagePart | MediaMessagePart]

    def render(self) -> LLMMessage:
        return {"role": "user", "content": [part.render() for part in self.parts]}


class AssistantMessage(BaseModel):
    text: str

    def render(self) -> LLMMessage:
        return {"role": "assistant", "content": [{"type": "text", "text": self.text}]}
