"""Goose: A framework for building LLM-based agents and workflows.

Goose provides tools for creating structured agent applications with support for:
- Task-based workflow orchestration
- Caching and state management
- Result validation and typing
- Agent conversations and refinement

Main components:
- Agent: Base agent for interacting with LLMs
- flow: Decorator for creating connected workflows
- task: Decorator for creating individual tasks
- Result: Base class for structured response types
"""

from goose._internal.agent import Agent
from goose._internal.flow import FlowArguments, flow
from goose._internal.result import Result, TextResult
from goose._internal.task import task
from goose._internal.types.telemetry import AgentResponse

__all__ = ["Agent", "flow", "FlowArguments", "Result", "TextResult", "task", "AgentResponse"]
