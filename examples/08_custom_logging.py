"""
Example demonstrating custom logging.

This example shows how to implement custom loggers to track
metrics about LLM interactions including tokens usage and timing.
"""

import asyncio
import csv
import os
import time
from datetime import datetime
from typing import Any

from aikernel import LLMMessagePart, LLMSystemMessage, LLMUserMessage, get_router
from pydantic import Field

from goose import Agent, AgentResponse, FlowArguments, Result, TextResult, flow, task
from goose._internal.agent import IAgentLogger


class CustomCSVLogger(IAgentLogger):
    """Custom logger that records LLM interactions to a CSV file."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.file_exists = os.path.exists(log_file)
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    async def __call__(self, *, response: AgentResponse[Any]) -> None:
        """Log the agent response to a CSV file."""
        # Calculate completion time
        completion_time = (response.end_time - response.start_time).total_seconds()
        
        # Extract the first few characters of input messages for brevity
        input_preview = str(response.input_messages)[:50].replace("\n", " ") + "..."
        
        # Prepare row data
        row = {
            'timestamp': datetime.now().isoformat(),
            'run_id': response.run_id,
            'flow_name': response.flow_name,
            'task_name': response.task_name,
            'model': response.model,
            'input_tokens': response.input_tokens,
            'output_tokens': response.output_tokens,
            'completion_time_seconds': f"{completion_time:.2f}",
            'input_preview': input_preview,
        }
        
        # Write to CSV file
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            
            # Write header if file is new
            if not self.file_exists:
                writer.writeheader()
                self.file_exists = True
            
            writer.writerow(row)
        
        # Also print to console
        print(f"Logged {response.task_name} - {response.input_tokens} input tokens, " +
              f"{response.output_tokens} output tokens, {completion_time:.2f}s")


class QuizQuestion(Result):
    question: str = Field(description="The quiz question")
    options: list[str] = Field(description="Multiple choice options")
    correct_option: int = Field(description="Index of the correct option (0-based)")
    explanation: str = Field(description="Explanation of the correct answer")


class LoggingFlowArguments(FlowArguments):
    topic: str
    difficulty: str


# Create a custom logger
logger = CustomCSVLogger(log_file="./logs/llm_interactions.csv")


@task
async def generate_quiz_question(*, agent: Agent, topic: str, difficulty: str) -> QuizQuestion:
    """Generate a quiz question on a specific topic."""
    # Create a router for Gemini 2.0 Flash
    router = get_router(models=("gemini-2.0-flash",))
    
    # System message with instructions
    system_message = LLMSystemMessage(
        parts=[LLMMessagePart(content="You are a helpful quiz creator.")]
    )
    
    # User request message
    user_message = LLMUserMessage(
        parts=[LLMMessagePart(content=f"Create a {difficulty} difficulty quiz question about {topic}. Include a question, 4 multiple choice options, the index of the correct option (0-based), and an explanation.")]
    )
    
    return await agent(
        messages=[system_message, user_message],
        model="gemini-2.0-flash",
        task_name="generate_quiz_question",
        response_model=QuizQuestion,
        router=router
    )


@task
async def generate_hint(*, agent: Agent, question: QuizQuestion) -> TextResult:
    """Generate a hint for a quiz question."""
    # Create a router for Gemini 2.0 Flash
    router = get_router(models=("gemini-2.0-flash",))
    
    # System message with instructions
    system_message = LLMSystemMessage(
        parts=[LLMMessagePart(content="You are a helpful tutor providing hints.")]
    )
    
    # User request message
    user_message = LLMUserMessage(
        parts=[LLMMessagePart(content=f"""
            Provide a subtle hint for this question without giving away the answer:
            
            Question: {question.question}
            Options: {', '.join([f"{i+1}. {option}" for i, option in enumerate(question.options)])}
        """)]
    )
    
    return await agent(
        messages=[system_message, user_message],
        model="gemini-2.0-flash",
        task_name="generate_hint",
        response_model=TextResult,
        router=router
    )


# Define a flow with the custom logger
@flow(agent_logger=logger)
async def quiz_generation_flow(*, flow_arguments: LoggingFlowArguments, agent: Agent) -> None:
    """Flow for generating quiz questions with logged metrics."""
    # Generate a quiz question
    question = await generate_quiz_question(
        agent=agent,
        topic=flow_arguments.topic,
        difficulty=flow_arguments.difficulty
    )
    
    # Generate a hint for the question
    await generate_hint(
        agent=agent,
        question=question
    )


async def generate_multiple_questions():
    """Generate multiple quiz questions to demonstrate logging."""
    topics = ["Python Programming", "Machine Learning", "World History", "Physics"]
    difficulties = ["easy", "medium", "hard", "expert"]
    
    # Create logs directory
    os.makedirs("./logs", exist_ok=True)
    
    for i, (topic, difficulty) in enumerate(zip(topics, difficulties)):
        print(f"\nGenerating question {i+1} about {topic} ({difficulty} difficulty)...")
        
        # Run with unique ID
        run_id = f"quiz-{int(time.time())}-{i}"
        
        async with quiz_generation_flow.start_run(run_id=run_id) as run:
            await quiz_generation_flow.generate(
                LoggingFlowArguments(
                    topic=topic,
                    difficulty=difficulty
                )
            )
            
            # Display the generated question
            question = run.get_result(task=generate_quiz_question)
            hint = run.get_result(task=generate_hint)
            
            print(f"Q: {question.question}")
            for i, option in enumerate(question.options):
                print(f"  {i+1}. {option}")
            print(f"Hint: {hint.text}")
            print(f"Answer: Option {question.correct_option + 1}")
            print(f"Explanation: {question.explanation}")


async def main():
    await generate_multiple_questions()
    
    # Display logging summary
    if os.path.exists("./logs/llm_interactions.csv"):
        print("\nLOG SUMMARY:")
        with open("./logs/llm_interactions.csv", 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # Calculate totals
            total_input_tokens = sum(int(row['input_tokens']) for row in rows)
            total_output_tokens = sum(int(row['output_tokens']) for row in rows)
            total_time = sum(float(row['completion_time_seconds']) for row in rows)
            
            print(f"Total interactions: {len(rows)}")
            print(f"Total input tokens: {total_input_tokens}")
            print(f"Total output tokens: {total_output_tokens}")
            print(f"Total completion time: {total_time:.2f}s")
            print(f"Average tokens per request: {(total_input_tokens + total_output_tokens) / len(rows):.1f}")
            print(f"Average completion time: {total_time / len(rows):.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
