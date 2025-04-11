"""
Example demonstrating iterative refinement.

This example shows how to refine a structured result by providing
feedback and using the refine method to improve it.
"""

import asyncio
from typing import List

from aikernel import LLMModelAlias, LLMRouter, LLMSystemMessage, LLMUserMessage
from pydantic import Field

from goose import Agent, FlowArguments, Result, flow, task


class CodeSolutionResult(Result):
    problem_understanding: str = Field(description="Understanding of the coding problem")
    solution: str = Field(description="Python code solution to the problem")
    explanation: str = Field(description="Explanation of how the solution works")
    time_complexity: str = Field(description="Time complexity analysis of the solution")
    space_complexity: str = Field(description="Space complexity analysis of the solution")


class RefineFlowArguments(FlowArguments):
    problem_statement: str


@task
async def generate_code_solution(*, agent: Agent, problem: str) -> CodeSolutionResult:
    """Generate a coding solution for a given problem."""
    router = LLMRouter[LLMModelAlias](
        model_list=[{"model_name": "gemini-2.0-flash", "litellm_params": {"model": "gemini/gemini-2.0-flash"}}],
        fallbacks=[],
    )
    
    return await agent(
        messages=[
            LLMSystemMessage(content="You are an expert coding assistant."),
            LLMUserMessage(content=f"Provide a solution to this coding problem:\n\n{problem}")
        ],
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
    problem = """
    Write a function that takes a list of integers and returns the two numbers that add up to a specific target.
    Assume there is exactly one solution, and you may not use the same element twice.
    Example:
    Input: nums = [2, 7, 11, 15], target = 9
    Output: [0, 1] (because nums[0] + nums[1] = 2 + 7 = 9)
    """
    
    # Start a flow run
    async with code_solution_flow.start_run(run_id="code-solution-refine") as run:
        # Generate initial solution
        await code_solution_flow.generate(RefineFlowArguments(problem_statement=problem))
        
        # Get the initial solution
        solution_task = run.get_task_instance(task=generate_code_solution)
        initial_solution = run.get_result(task=generate_code_solution)
        
        print("INITIAL SOLUTION:")
        print("=" * 50)
        print(f"Problem Understanding:\n{initial_solution.problem_understanding}\n")
        print(f"Solution Code:\n{initial_solution.solution}\n")
        print(f"Explanation:\n{initial_solution.explanation}\n")
        print(f"Time Complexity: {initial_solution.time_complexity}")
        print(f"Space Complexity: {initial_solution.space_complexity}")
        print("=" * 50)
        
        # First refinement: Optimize time complexity
        print("\nREFINING: Optimizing time complexity...")
        optimized_solution = await solution_task.refine(
            user_message=LLMUserMessage(content="Please optimize the solution for better time complexity."),
            model="gemini-2.0-flash",
            router=LLMRouter[LLMModelAlias](
                model_list=[{"model_name": "gemini-2.0-flash", "litellm_params": {"model": "gemini/gemini-2.0-flash"}}],
                fallbacks=[],
            )
        )
        
        print("\nOPTIMIZED SOLUTION:")
        print("=" * 50)
        print(f"Problem Understanding:\n{optimized_solution.problem_understanding}\n")
        print(f"Solution Code:\n{optimized_solution.solution}\n")
        print(f"Explanation:\n{optimized_solution.explanation}\n")
        print(f"Time Complexity: {optimized_solution.time_complexity}")
        print(f"Space Complexity: {optimized_solution.space_complexity}")
        print("=" * 50)
        
        # Second refinement: Add test cases
        print("\nREFINING: Adding test cases...")
        final_solution = await solution_task.refine(
            user_message=LLMUserMessage(content="Please add test cases to verify the solution works correctly."),
            model="gemini-2.0-flash",
            router=LLMRouter[LLMModelAlias](
                model_list=[{"model_name": "gemini-2.0-flash", "litellm_params": {"model": "gemini/gemini-2.0-flash"}}],
                fallbacks=[],
            )
        )
        
        print("\nFINAL SOLUTION WITH TESTS:")
        print("=" * 50)
        print(f"Problem Understanding:\n{final_solution.problem_understanding}\n")
        print(f"Solution Code:\n{final_solution.solution}\n")
        print(f"Explanation:\n{final_solution.explanation}\n")
        print(f"Time Complexity: {final_solution.time_complexity}")
        print(f"Space Complexity: {final_solution.space_complexity}")
        print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())