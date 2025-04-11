"""
Example demonstrating stateful conversations.

This example shows how to maintain conversation history and context
across multiple interactions with a task.
"""

import asyncio

from aikernel import LLMModelAlias, LLMRouter, LLMSystemMessage, LLMUserMessage

from goose import Agent, FlowArguments, TextResult, flow, task


class TutorFlowArguments(FlowArguments):
    subject: str
    student_level: str


@task
async def explain_concept(*, agent: Agent, concept: str) -> TextResult:
    """Explain a concept and maintain conversation history."""
    router = LLMRouter[LLMModelAlias](
        model_list=[{"model_name": "gemini-2.0-flash", "litellm_params": {"model": "gemini/gemini-2.0-flash"}}],
        fallbacks=[],
    )
    
    return await agent(
        messages=[
            LLMSystemMessage(content="You are a helpful educational tutor."),
            LLMUserMessage(content=f"Explain this concept: {concept}")
        ],
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
    # Start a flow run for the tutoring session
    async with tutoring_flow.start_run(run_id="tutoring-session-1") as run:
        # Initialize the flow
        await tutoring_flow.generate(
            TutorFlowArguments(
                subject="Physics",
                student_level="Undergraduate"
            )
        )
        
        # Get the explain_concept task
        explanation_task = run.get_task_instance(task=explain_concept)
        
        # Get the initial explanation
        initial_explanation = run.get_result(task=explain_concept)
        print("INITIAL EXPLANATION:")
        print(initial_explanation.text)
        print("\n" + "-" * 50 + "\n")
        
        # Ask follow-up questions using the same conversation context
        follow_up_questions = [
            "Can you explain it in simpler terms?",
            "How is this related to quantum computing?",
            "What are some real-world applications?"
        ]
        
        for i, question in enumerate(follow_up_questions, 1):
            # Use the ask method to continue the conversation with the same context
            response = await explanation_task.ask(
                user_message=LLMUserMessage(content=question),
                model="gemini-2.0-flash",
                router=LLMRouter[LLMModelAlias](
                    model_list=[{"model_name": "gemini-2.0-flash", "litellm_params": {"model": "gemini/gemini-2.0-flash"}}],
                    fallbacks=[],
                )
            )
            
            print(f"FOLLOW-UP QUESTION {i}: {question}")
            print(response)
            print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    asyncio.run(main())