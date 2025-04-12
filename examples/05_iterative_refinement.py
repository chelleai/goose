"""
Example demonstrating iterative refinement.

This example shows how to refine a structured result by providing
feedback and using the refine method to improve it.
"""

import asyncio
import os

from aikernel import LLMMessagePart, LLMSystemMessage, LLMUserMessage, get_router
from pydantic import Field

from goose import Agent, FlowArguments, Result, flow, task


class CodeSolutionResult(Result):
    """Structured solution to a coding problem."""
    problem_understanding: str = Field(description="Understanding of the coding problem")
    solution: str = Field(description="Python code solution to the problem")
    explanation: str = Field(description="Explanation of how the solution works")
    time_complexity: str = Field(description="Time complexity analysis of the solution")
    space_complexity: str = Field(description="Space complexity analysis of the solution")


class RefineFlowArguments(FlowArguments):
    """Arguments for the code solution flow."""
    problem_statement: str




@task
async def generate_code_solution(*, agent: Agent, problem: str) -> CodeSolutionResult:
    """Generate a coding solution for a given problem."""
    print(f"Generating initial solution for problem...")
    
    # Create a router for Gemini 2.0 Flash
    router = get_router(models=("gemini-2.0-flash",))
    
    # System message with instructions
    system_message = LLMSystemMessage(
        parts=[LLMMessagePart(content="You are an expert programmer. Create a solution to the given coding problem with detailed explanation.")]
    )
    
    # User request message
    user_message = LLMUserMessage(
        parts=[LLMMessagePart(content=f"Please solve this coding problem:\n\n{problem}\n\nProvide your understanding of the problem, a Python code solution, explanation of how it works, and time/space complexity analysis.")]
    )
    
    # Make the actual LLM call
    return await agent(
        messages=[system_message, user_message],
        model="gemini-2.0-flash",
        task_name="generate_code_solution",
        response_model=CodeSolutionResult,
        router=router
    )


@flow
async def code_solution_flow(*, flow_arguments: RefineFlowArguments, agent: Agent) -> None:
    """Flow for generating and refining code solutions."""
    await generate_code_solution(agent=agent, problem=flow_arguments.problem_statement)


async def main():
    """Run the code solution flow and demonstrate iterative refinement."""
    # Create a unique run ID
    run_id = f"code-solution-{os.getpid()}"
    
    print("=== Iterative Refinement Example ===")
    print("This example demonstrates how to refine a structured result")
    print("by providing feedback and using the refine method to improve it.\n")
    
    problem = """
    Write a function that takes a list of integers and returns the two numbers that add up to a specific target.
    Assume there is exactly one solution, and you may not use the same element twice.
    Example:
    Input: nums = [2, 7, 11, 15], target = 9
    Output: [0, 1] (because nums[0] + nums[1] = 2 + 7 = 9)
    """
    
    # Start a flow run
    async with code_solution_flow.start_run(run_id=run_id) as run:
        # Generate initial solution
        await code_solution_flow.generate(RefineFlowArguments(problem_statement=problem))
        
        # Get the initial solution
        initial_solution_result = run.get_result(task=generate_code_solution)
        
        print("\n--- Initial Solution ---")
        print("=" * 50)
        print(f"Problem Understanding:\n{initial_solution_result.problem_understanding}\n")
        print(f"Solution Code:\n{initial_solution_result.solution}\n")
        print(f"Explanation:\n{initial_solution_result.explanation}\n")
        print(f"Time Complexity: {initial_solution_result.time_complexity}")
        print(f"Space Complexity: {initial_solution_result.space_complexity}")
        print("=" * 50)
        
        # Create a router for refinements
        router = get_router(models=("gemini-2.0-flash",))
        
        # First refinement: Optimize time complexity
        print("\n--- First Refinement: Optimizing time complexity ---")
        print("User feedback: \"Please optimize the solution for better time complexity.\"")
        
        # Create a user message for the refinement
        user_message = LLMUserMessage(
            parts=[LLMMessagePart(content="Please optimize the solution for better time complexity.")]
        )
        
        # Use the task's refine method to improve the solution
        await generate_code_solution.refine(
            user_message=user_message,
            model="gemini-2.0-flash",
            router=router,
            index=0  # Use the first instance of the task
        )
        
        # Get the optimized solution
        optimized_solution_result = run.get_result(task=generate_code_solution)
        
        print("\n--- Optimized Solution ---")
        print("=" * 50)
        print(f"Problem Understanding:\n{optimized_solution_result.problem_understanding}\n")
        print(f"Solution Code:\n{optimized_solution_result.solution}\n")
        print(f"Explanation:\n{optimized_solution_result.explanation}\n")
        print(f"Time Complexity: {optimized_solution_result.time_complexity}")
        print(f"Space Complexity: {optimized_solution_result.space_complexity}")
        print("=" * 50)
        
        # Second refinement: Add test cases
        print("\n--- Second Refinement: Adding test cases ---")
        print("User feedback: \"Please add test cases to verify the solution works correctly.\"")
        
        # Create a user message for the second refinement
        user_message = LLMUserMessage(
            parts=[LLMMessagePart(content="Please add test cases to verify the solution works correctly.")]
        )
        
        # Use the task's refine method again to add test cases
        await generate_code_solution.refine(
            user_message=user_message,
            model="gemini-2.0-flash",
            router=router,
            index=0  # Use the first instance of the task
        )
        
        # Get the final solution with test cases
        final_solution_result = run.get_result(task=generate_code_solution)
        
        print("\n--- Final Solution With Tests ---")
        print("=" * 50)
        print(f"Problem Understanding:\n{final_solution_result.problem_understanding}\n")
        print(f"Solution Code:\n{final_solution_result.solution}\n")
        print(f"Explanation:\n{final_solution_result.explanation}\n")
        print(f"Time Complexity: {final_solution_result.time_complexity}")
        print(f"Space Complexity: {final_solution_result.space_complexity}")
        print("=" * 50)
        
        print("\nNOTE: The refine method allows for iterative improvement of results based on")
        print("user feedback, while maintaining the same structured output format.")


if __name__ == "__main__":
    asyncio.run(main())
