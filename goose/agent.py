import logging
from datetime import datetime
from typing import Any, Callable

from litellm import acompletion
from pydantic import BaseModel

from goose.types.agent import (
    AgentResponse,
    AssistantMessage,
    GeminiModel,
    SystemMessage,
    UserMessage,
)


class Agent:
    def __init__(
        self,
        *,
        flow_name: str,
        run_id: str,
        logger: Callable[[AgentResponse[Any]], None] | None = None,
    ) -> None:
        self.flow_name = flow_name
        self.run_id = run_id
        self.logger = logger or logging.info

    async def __call__[R: BaseModel](
        self,
        *,
        messages: list[UserMessage | AssistantMessage],
        model: GeminiModel,
        response_model: type[R],
        task_name: str,
        system: SystemMessage | None = None,
    ) -> R:
        start_time = datetime.now()
        rendered_messages = [message.render() for message in messages]
        if system is not None:
            rendered_messages.insert(0, system.render())

        response = await acompletion(
            model=model.value,
            messages=rendered_messages,
            response_format={
                "type": "json_object",
                "response_schema": response_model.model_json_schema(),
                "enforce_validation": True,
            },
        )

        if len(response.choices) == 0:
            raise RuntimeError("No content returned from LLM call.")

        parsed_response = response_model.model_validate_json(
            response.choices[0].message.content
        )
        end_time = datetime.now()
        agent_response = AgentResponse(
            response=parsed_response,
            id=self.run_id,
            flow_name=self.flow_name,
            task_name=task_name,
            model=model,
            system=system,
            input_messages=messages,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            start_time=start_time,
            end_time=end_time,
        )

        self.logger(agent_response)
        return agent_response.response
