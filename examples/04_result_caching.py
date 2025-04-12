"""
Example demonstrating result caching.

This example shows how Goose automatically caches results based on
input hashing and only regenerates results when inputs change.

NOTE: This is a demo file that illustrates the pattern without making actual LLM calls.
"""

import asyncio
import os
import time
from typing import List, Dict

from pydantic import Field

from goose import Agent, FlowArguments, Result, flow, task


class SummaryResult(Result):
    """Structured summary of a text."""
    summary: str = Field(description="Summary of the text")
    key_points: List[str] = Field(description="Key points from the text")
    suggested_title: str = Field(description="A suggested title for the text")


class CacheFlowArguments(FlowArguments):
    """Arguments for the summary flow."""
    content: str


# Mock implementation
def mock_summary_generator(text: str) -> SummaryResult:
    """Generate a mock summary based on the content."""
    # Simple way to identify which text we're summarizing
    if "Artificial Intelligence" in text or "AI" in text:
        return SummaryResult(
            summary="Artificial Intelligence has transformed industries through machine learning and deep learning, offering benefits while raising concerns about privacy, bias, and job displacement that researchers and policymakers are working to address.",
            key_points=[
                "AI has evolved rapidly with advances in machine learning and deep learning",
                "Applications include NLP, computer vision, and predictive analytics",
                "Companies are adopting AI to optimize operations and improve customer experience",
                "Concerns include privacy issues, algorithmic bias, and workforce disruption",
                "Researchers and policymakers are working to address these challenges"
            ],
            suggested_title="The Rise of AI: Opportunities and Challenges"
        )
    elif "Quantum computing" in text or "qubits" in text:
        return SummaryResult(
            summary="Quantum computing uses qubits in superposition to solve complex problems exponentially faster than classical computers, with potential applications in cryptography, drug discovery, material science, and optimization problems, despite still being in early development.",
            key_points=[
                "Quantum computers use qubits instead of binary bits",
                "Superposition allows qubits to exist in multiple states simultaneously",
                "Quantum computers can solve certain problems exponentially faster",
                "Applications include cryptography, drug discovery, and optimization",
                "Major companies and research institutions are investing in quantum technology"
            ],
            suggested_title="Quantum Computing: The Next Frontier in Computation"
        )
    else:
        return SummaryResult(
            summary=f"Summary of text ({len(text)} characters)",
            key_points=["Generic key point 1", "Generic key point 2"],
            suggested_title="Generic Title"
        )


# Dictionary to simulate cache behavior for demonstration
mock_cache: Dict[str, bool] = {}


@task
async def summarize_text(*, agent: Agent, text: str) -> SummaryResult:
    """Summarize the given text.
    
    In a real implementation, this would call an LLM through the agent.
    For this example, we use a mock implementation to demonstrate caching.
    """
    # Create a simple hash of the input for our mock cache
    text_hash = hash(text) % 10000
    
    # Check if we've "processed" this text before
    if text_hash in mock_cache:
        print(f"Cache hit for text hash {text_hash} - Using cached result.")
    else:
        print(f"Cache miss for text hash {text_hash} - Generating new summary for text... (length: {len(text)} chars)")
        # Add a delay to simulate LLM processing time
        time.sleep(1)
        # In a real implementation, this would make an actual API call
        mock_cache[text_hash] = True
    
    # In a real implementation, you would use the agent to call the LLM:
    # return await agent(
    #     messages=[...],
    #     model="your-model",
    #     task_name="summarize_text",
    #     response_model=SummaryResult
    # )
    
    # For this example, we use a mock implementation
    return mock_summary_generator(text)


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