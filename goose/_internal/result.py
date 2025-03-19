from pydantic import BaseModel, ConfigDict, Field


class Result(BaseModel):
    model_config = ConfigDict(frozen=True)


class TextResult(Result):
    text: str


class Replacement(BaseModel):
    find: str = Field(description="Text to find, to be replaced with `replace`")
    replace: str = Field(description="Text to replace `find` with")


class FindReplaceResponse(BaseModel):
    replacements: list[Replacement] = Field(
        description="List of replacements to make in the previous result to satisfy the user's request"
    )
