from pydantic import BaseModel, ConfigDict


class Result(BaseModel):
    model_config = ConfigDict(frozen=True)


class TextResult(Result):
    text: str
