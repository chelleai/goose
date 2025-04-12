"""
Example demonstrating iterative refinement.

This example shows how to refine a structured result by providing
feedback and using the refine method to improve it.

NOTE: This is a demo file that illustrates the pattern without making actual LLM calls.
"""

import asyncio
import os
from typing import List

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


# Mock implementations for our solutions
def initial_solution() -> CodeSolutionResult:
    """Generate an initial solution with brute force approach."""
    return CodeSolutionResult(
        problem_understanding="This problem asks us to find two indices in an array such that their corresponding values add up to a target value. We're guaranteed that a solution exists, and we can't use the same element twice.",
        solution="""def two_sum(nums, target):
    # Brute force approach: check all possible pairs
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    # If no solution is found (shouldn't happen per problem constraints)
    return []""",
        explanation="This solution uses a simple brute force approach. We use nested loops to check every possible pair of numbers in the array. For each pair, we check if they add up to the target. If they do, we return their indices.",
        time_complexity="O(n²) where n is the length of the input array",
        space_complexity="O(1) as we only use a constant amount of extra space"
    )


def optimized_solution() -> CodeSolutionResult:
    """Generate an optimized solution using hash map."""
    return CodeSolutionResult(
        problem_understanding="This problem asks us to find two indices in an array such that their corresponding values add up to a target value. We're guaranteed that a solution exists, and we can't use the same element twice.",
        solution="""def two_sum(nums, target):
    # Use a hash map to store seen numbers and their indices
    seen = {}
    
    for i, num in enumerate(nums):
        # Calculate complement (the value we need to find)
        complement = target - num
        
        # If complement is in the hash map, we found our solution
        if complement in seen:
            return [seen[complement], i]
            
        # Otherwise, add current number and its index to the hash map
        seen[num] = i
        
    # Should not reach here based on problem constraints
    return []""",
        explanation="This solution uses a hash map to achieve O(n) time complexity. We iterate through the array once, and for each element, we check if its complement (target - current_number) is in our hash map. If it is, we've found our solution. If not, we add the current number and its index to our hash map and continue.",
        time_complexity="O(n) where n is the length of the input array",
        space_complexity="O(n) as we might need to store up to n elements in the hash map"
    )


def solution_with_tests() -> CodeSolutionResult:
    """Generate the final solution with test cases."""
    return CodeSolutionResult(
        problem_understanding="This problem asks us to find two indices in an array such that their corresponding values add up to a target value. We're guaranteed that a solution exists, and we can't use the same element twice.",
        solution="""def two_sum(nums, target):
    # Use a hash map to store seen numbers and their indices
    seen = {}
    
    for i, num in enumerate(nums):
        # Calculate complement (the value we need to find)
        complement = target - num
        
        # If complement is in the hash map, we found our solution
        if complement in seen:
            return [seen[complement], i]
            
        # Otherwise, add current number and its index to the hash map
        seen[num] = i
        
    # Should not reach here based on problem constraints
    return []


# Test cases
def run_tests():
    test_cases = [
        {"nums": [2, 7, 11, 15], "target": 9, "expected": [0, 1]},
        {"nums": [3, 2, 4], "target": 6, "expected": [1, 2]},
        {"nums": [3, 3], "target": 6, "expected": [0, 1]},
        {"nums": [1, 5, 8, 10, 13], "target": 18, "expected": [2, 4]},
        {"nums": [-1, -2, -3, -4, -5], "target": -8, "expected": [2, 4]}
    ]
    
    for i, test in enumerate(test_cases):
        result = two_sum(test["nums"], test["target"])
        assert result == test["expected"], f"Test {i+1} failed: got {result}, expected {test['expected']}"
        print(f"Test {i+1} passed!")
    
    print("All tests passed!")


if __name__ == "__main__":
    run_tests()""",
        explanation="This solution uses a hash map to achieve O(n) time complexity. We iterate through the array once, and for each element, we check if its complement (target - current_number) is in our hash map. If it is, we've found our solution. If not, we add the current number and its index to our hash map and continue. I've also added comprehensive test cases that cover various scenarios including positive numbers, negative numbers, and duplicate values.",
        time_complexity="O(n) where n is the length of the input array",
        space_complexity="O(n) as we might need to store up to n elements in the hash map"
    )


@task
async def generate_code_solution(*, agent: Agent, problem: str) -> CodeSolutionResult:
    """Generate a coding solution for a given problem.
    
    In a real implementation, this would call an LLM through the agent.
    For this example, we use a mock implementation.
    """
    print(f"Generating initial solution for problem...")
    
    # In a real implementation, you would use the agent to call the LLM:
    # return await agent(
    #     messages=[...],
    #     model="your-model",
    #     task_name="generate_code_solution",
    #     response_model=CodeSolutionResult
    # )
    
    # For this example, we use a mock implementation
    return initial_solution()


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
        
        # In a real implementation, we would use task.refine:
        # optimized_solution = await generate_code_solution.refine(
        #     user_message=LLMUserMessage(...),
        #     model="your-model",
        #     router=router
        # )
        
        # For demonstration, we'll use our mock implementations
        
        # First refinement: Optimize time complexity
        print("\n--- First Refinement: Optimizing time complexity ---")
        print("User feedback: \"Please optimize the solution for better time complexity.\"")
        
        # In the real implementation, this would be the result of the refine method
        optimized_solution_result = optimized_solution()
        
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
        
        # In the real implementation, this would be the result of the refine method
        final_solution_result = solution_with_tests()
        
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