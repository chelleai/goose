from enum import StrEnum
from typing import NewType

from pydantic import BaseModel

EncodedMediaMessage = NewType("EncodedMediaMessage", str)


class SupportedMimeType(StrEnum):
    PDF = "application/pdf"
    HTML = "text/html"
    PLAIN = "text/plain"
    CSV = "text/csv"
    PNG = "image/png"
    JPEG = "image/jpeg"
    WEBP = "image/webp"
    WAV = "audio/wav"
    MP3 = "audio/mp3"
    AAC = "audio/aac"
    FLAC = "audio/flac"
    DOCX = "application/msword"


class MediaMessage(BaseModel):
    mime_type: SupportedMimeType
    data: bytes
