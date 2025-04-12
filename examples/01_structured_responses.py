"""
Example demonstrating structured LLM responses.

This example shows how to create a structured result type and use it
with a flow to ensure the LLM output conforms to expected schema.
"""

import asyncio
import os
from typing import List

from aikernel import LLMMessagePart, LLMRouter, LLMSystemMessage, LLMUserMessage, get_router
from pydantic import Field

from goose import Agent, FlowArguments, Result, flow, task


# Define a structured result type
class RecipeResult(Result):
    """Recipe with structured attributes."""
    title: str = Field(description="The title of the recipe")
    ingredients: List[str] = Field(description="List of ingredients needed")
    steps: List[str] = Field(description="Step-by-step cooking instructions")
    prep_time_minutes: int = Field(description="Preparation time in minutes")
    cooking_time_minutes: int = Field(description="Cooking time in minutes")


class RecipeFlowArguments(FlowArguments):
    """Arguments for the recipe flow."""
    ingredient: str


@task
async def generate_recipe(*, agent: Agent, ingredient: str) -> RecipeResult:
    """Generate a recipe that uses the specified ingredient."""
    print(f"Generating recipe for {ingredient}...")
    
    # Create a router for Gemini 2.0 Flash
    router = get_router(models=("gemini-2.0-flash",))
    
    # System message with instructions
    system_message = LLMSystemMessage(
        parts=[LLMMessagePart(content=f"You are a creative chef. Create a recipe using {ingredient} as a main ingredient.")]
    )
    
    # User request message
    user_message = LLMUserMessage(
        parts=[LLMMessagePart(content=f"Please create a recipe that features {ingredient} as a main ingredient.")]
    )
    
    # Make the actual LLM call
    return await agent(
        messages=[system_message, user_message],
        model="gemini-2.0-flash",
        task_name="generate_recipe",
        response_model=RecipeResult,
        router=router
    )


@flow
async def recipe_flow(*, flow_arguments: RecipeFlowArguments, agent: Agent) -> None:
    """Flow for generating a recipe with structured output."""
    await generate_recipe(agent=agent, ingredient=flow_arguments.ingredient)


async def main():
    """Run the recipe flow and display the results."""
    # Create a unique run ID
    run_id = f"recipe-{os.getpid()}"
    
    print("=== Structured LLM Responses Example ===")
    print("This example demonstrates how to create structured result types")
    print("and ensure LLM outputs conform to expected schemas.\n")
    
    # Run the recipe flow
    async with recipe_flow.start_run(run_id=run_id) as run:
        await recipe_flow.generate(RecipeFlowArguments(ingredient="avocado"))
        
        # Get the recipe from the result
        recipe = run.get_result(task=generate_recipe)
        
        # Display the recipe information
        print("\n--- Generated Recipe ---")
        print(f"Recipe: {recipe.title}")
        print("\nIngredients:")
        for item in recipe.ingredients:
            print(f"- {item}")
        
        print("\nInstructions:")
        for i, step in enumerate(recipe.steps, 1):
            print(f"{i}. {step}")
        
        print(f"\nPrep time: {recipe.prep_time_minutes} minutes")
        print(f"Cooking time: {recipe.cooking_time_minutes} minutes")
        
        # Access fields directly with type safety
        total_time = recipe.prep_time_minutes + recipe.cooking_time_minutes
        print(f"Total time: {total_time} minutes")
        
        print("\nNote: This demonstrates how structured outputs provide type safety and")
        print("predictable fields that can be accessed in your application code.")


if __name__ == "__main__":
    asyncio.run(main())
