"""
Example demonstrating stateful conversations.

This example shows how to maintain conversation history and context
across multiple interactions with a task.
"""

import asyncio
import os

from aikernel import LLMMessagePart, LLMSystemMessage, LLMUserMessage, get_router
from pydantic import Field

from goose import Agent, FlowArguments, TextResult, flow, task


class TutorFlowArguments(FlowArguments):
    """Arguments for the tutoring flow."""
    subject: str
    student_level: str




@task
async def explain_concept(*, agent: Agent, concept: str) -> TextResult:
    """Explain a concept and maintain conversation history."""
    print(f"Generating explanation for concept: {concept}")
    
    # Create a router for Gemini 2.0 Flash
    router = get_router(models=("gemini-2.0-flash",))
    
    # System message with instructions
    system_message = LLMSystemMessage(
        parts=[LLMMessagePart(content=f"You are a knowledgeable tutor. Explain the concept of {concept} clearly and accurately.")]
    )
    
    # User request message
    user_message = LLMUserMessage(
        parts=[LLMMessagePart(content=f"Please explain {concept} in detail. Include both a technical explanation and some context about its importance.")]
    )
    
    # Make the actual LLM call
    return await agent(
        messages=[system_message, user_message],
        model="gemini-2.0-flash",
        task_name="explain_concept",
        response_model=TextResult,
        router=router
    )


@flow
async def tutoring_flow(*, flow_arguments: TutorFlowArguments, agent: Agent) -> None:
    """Flow for a tutoring session that maintains conversation history."""
    # Initial explanation
    await explain_concept(agent=agent, concept="quantum entanglement")


async def main():
    """Run the tutoring flow and demonstrate stateful conversations."""
    # Create a unique run ID
    run_id = f"tutoring-{os.getpid()}"
    
    print("=== Stateful Conversations Example ===")
    print("This example demonstrates how to maintain conversation history")
    print("across multiple interactions with a task.\n")
    
    # Start a flow run for the tutoring session
    async with tutoring_flow.start_run(run_id=run_id) as run:
        # Initialize the flow
        await tutoring_flow.generate(
            TutorFlowArguments(
                subject="Physics",
                student_level="Undergraduate"
            )
        )
        
        # Get the initial explanation
        initial_explanation = run.get_result(task=explain_concept)
        print("\n--- Initial Explanation ---")
        print(initial_explanation.text)
        print("\n" + "-" * 50)
        
        # Ask follow-up questions using the same conversation context
        follow_up_questions = [
            "Can you explain it in simpler terms?",
            "How is this related to quantum computing?",
            "What are some real-world applications?"
        ]
        
        # Create a router for follow-up questions
        router = get_router(models=("gemini-2.0-flash",))
        
        # Ask follow-up questions using the task's ask method
        for i, question in enumerate(follow_up_questions, 1):
            print(f"\n--- Follow-up Question {i}: {question} ---")
            
            # Create a user message for the follow-up question
            user_message = LLMUserMessage(
                parts=[LLMMessagePart(content=question)]
            )
            
            # Use the task's ask method to maintain conversation context
            response = await explain_concept.ask(
                user_message=user_message,
                model="gemini-2.0-flash",
                router=router,
                index=0  # Use the first instance of the task
            )
            
            print(response)
            print("\nNOTE: The system has automatically maintained the conversation context.")
            print("The LLM received the full history of the conversation,")
            print("allowing it to provide coherent follow-up responses.")
            
            print("-" * 50)
        
        print("\nThis example demonstrates how Goose maintains conversation state")
        print("across multiple interactions, allowing for natural follow-up questions.")


if __name__ == "__main__":
    asyncio.run(main())
