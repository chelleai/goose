"""Result models for agent responses.

This module provides the base classes and models for structured results from LLM agents.
"""

from pydantic import BaseModel, ConfigDict, Field


class Result(BaseModel):
    """Base class for all result models.

    All result models in Goose extend from this class to have consistent
    behavior. Results are frozen (immutable) by default.
    """

    model_config = ConfigDict(frozen=True)


class TextResult(Result):
    """Simple text result model.

    A basic result type that contains unstructured text output from an agent.

    Attributes:
        text: The text content of the result
    """

    text: str


class Replacement(BaseModel):
    """Represents a text replacement operation.

    Used in find-and-replace operations to refine structured results.

    Attributes:
        find: The text to find in the original result
        replace: The text to replace the found text with
    """

    find: str = Field(description="Text to find, to be replaced with `replace`")
    replace: str = Field(description="Text to replace `find` with")


class FindReplaceResponse(BaseModel):
    """Model for find-and-replace operations on structured results.

    Used to refine existing results by applying a series of replacements.

    Attributes:
        replacements: List of replacement operations to apply
    """

    replacements: list[Replacement] = Field(
        description="List of replacements to make in the previous result to satisfy the user's request"
    )
