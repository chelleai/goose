"""
Example demonstrating result caching.

This example shows how Goose automatically caches results based on
input hashing and only regenerates results when inputs change.
"""

import asyncio
import time
from typing import List

from aikernel import LLMModelAlias, LLMRouter, LLMSystemMessage, LLMUserMessage
from pydantic import Field

from goose import Agent, FlowArguments, Result, flow, task


class SummaryResult(Result):
    summary: str = Field(description="Summary of the text")
    key_points: List[str] = Field(description="Key points from the text")
    suggested_title: str = Field(description="A suggested title for the text")


class CacheFlowArguments(FlowArguments):
    content: str


@task
async def summarize_text(*, agent: Agent, text: str) -> SummaryResult:
    """Summarize the given text."""
    router = LLMRouter[LLMModelAlias](
        model_list=[{"model_name": "gemini-2.0-flash", "litellm_params": {"model": "gemini/gemini-2.0-flash"}}],
        fallbacks=[],
    )
    
    print(f"Generating summary for text... (length: {len(text)} chars)")
    
    # Add a delay to clearly show when the LLM is actually called
    time.sleep(1)
    
    return await agent(
        messages=[
            LLMSystemMessage(content="You are a helpful summarizer."),
            LLMUserMessage(content=f"Summarize this text:\n\n{text}")
        ],
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
    
    async with summary_flow.start_run(run_id="summary-cache-demo") as run:
        print("\nRun 1: Initial summarization")
        await summary_flow.generate(CacheFlowArguments(content=article1))
        
        # Get and display the result
        summary1 = run.get_result(task=summarize_text)
        print(f"Title: {summary1.suggested_title}")
        print(f"Summary: {summary1.summary}")
        print("Key points:")
        for point in summary1.key_points:
            print(f"- {point}")
        
        print("\nRun 2: Same input (should use cached result)")
        await summary_flow.generate(CacheFlowArguments(content=article1))
        
        print("\nRun 3: Different input (should generate new result)")
        await summary_flow.generate(CacheFlowArguments(content=article2))
        
        # Get and display the new result
        summary2 = run.get_result(task=summarize_text)
        print(f"Title: {summary2.suggested_title}")
        print(f"Summary: {summary2.summary}")
        print("Key points:")
        for point in summary2.key_points:
            print(f"- {point}")


if __name__ == "__main__":
    asyncio.run(main())