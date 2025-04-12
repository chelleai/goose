"""
Example demonstrating result caching.

This example shows how Goose automatically caches results based on
input hashing and only regenerates results when inputs change.
"""

import asyncio
import os

from aikernel import LLMMessagePart, LLMSystemMessage, LLMUserMessage, get_router
from pydantic import Field

from goose import Agent, FlowArguments, Result, flow, task


class SummaryResult(Result):
    """Structured summary of a text."""
    summary: str = Field(description="Summary of the text")
    key_points: list[str] = Field(description="Key points from the text")
    suggested_title: str = Field(description="A suggested title for the text")


class CacheFlowArguments(FlowArguments):
    """Arguments for the summary flow."""
    content: str


@task
async def summarize_text(*, agent: Agent, text: str) -> SummaryResult:
    """Summarize the given text.
    
    This task demonstrates caching behavior - when called with the same text
    multiple times, it will only make one LLM call and reuse the result.
    """
    print(f"Processing text (length: {len(text)} chars)")
    
    # Create a router for Gemini 2.0 Flash
    router = get_router(models=("gemini-2.0-flash",))
    
    # System message with instructions
    system_message = LLMSystemMessage(
        parts=[LLMMessagePart(content="You are a helpful assistant that summarizes text. Create a concise summary with key points.")]
    )
    
    # User request message
    user_message = LLMUserMessage(
        parts=[LLMMessagePart(content=f"Please summarize the following text:\n\n{text}\n\nExtract the key points and provide a suggested title.")]
    )
    
    # Make the actual LLM call
    return await agent(
        messages=[system_message, user_message],
        model="gemini-2.0-flash",
        task_name="summarize_text",
        response_model=SummaryResult,
        router=router
    )


@flow
async def summary_flow(*, flow_arguments: CacheFlowArguments, agent: Agent) -> None:
    """Flow that demonstrates caching of task results."""
    await summarize_text(agent=agent, text=flow_arguments.content)


async def main():
    """Run the summary flow to demonstrate result caching."""
    # Create a unique run ID
    run_id = f"summary-{os.getpid()}"
    
    print("=== Result Caching Example ===")
    print("This example demonstrates how Goose automatically caches results")
    print("based on input hashing and only regenerates results when inputs change.\n")
    
    # Sample text to summarize
    article1 = """
    Artificial Intelligence (AI) has rapidly evolved in recent years, transforming various industries.
    Machine learning algorithms, particularly deep learning models, have made significant strides in 
    natural language processing, computer vision, and predictive analytics. Companies across sectors 
    are integrating AI solutions to optimize operations, enhance customer experiences, and gain competitive advantages.
    However, this technological boom also raises concerns about privacy, bias, and job displacement.
    Researchers and policymakers are working to address these challenges while harnessing AI's benefits.
    """
    
    article2 = """
    Quantum computing represents a revolutionary approach to computation. Unlike classical computers that
    use bits, quantum computers utilize quantum bits or qubits, which can exist in multiple states simultaneously
    due to superposition. This property enables quantum computers to solve certain complex problems exponentially
    faster than traditional computers. While still in early development stages, quantum computing has potential
    applications in cryptography, drug discovery, material science, and optimization problems.
    Major technology companies and research institutions continue to invest heavily in advancing quantum technologies.
    """
    
    # Start the flow
    async with summary_flow.start_run(run_id=run_id) as run:
        print("\n--- Run 1: Initial summarization ---")
        await summary_flow.generate(CacheFlowArguments(content=article1))
        
        # Get and display the result
        summary1 = run.get_result(task=summarize_text)
        print(f"\nTitle: {summary1.suggested_title}")
        print(f"Summary: {summary1.summary}")
        print("Key points:")
        for point in summary1.key_points:
            print(f"- {point}")
        
        print("\n--- Run 2: Same input (should use cached result) ---")
        await summary_flow.generate(CacheFlowArguments(content=article1))
        print("\nNOTE: The task used the cached result without making another LLM call")
        print("because the input hash matched the previous call.")
        
        print("\n--- Run 3: Different input (should generate new result) ---")
        await summary_flow.generate(CacheFlowArguments(content=article2))
        
        # Get and display the new result - this should be the quantum computing summary
        summary2 = run.get_result(task=summarize_text)
        print(f"\nTitle: {summary2.suggested_title}")
        print(f"Summary: {summary2.summary}")
        print("Key points:")
        for point in summary2.key_points:
            print(f"- {point}")
        
        print("\nNOTE: Behind the scenes, Goose has automatically cached the results based on")
        print("the input hash. When the same input is provided, it returns the cached result")
        print("without calling the LLM again, saving time and API costs.")


if __name__ == "__main__":
    asyncio.run(main())
