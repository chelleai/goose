"""
Example demonstrating structured LLM responses.

This example shows how to create a structured result type and use it
with a task to ensure the LLM output conforms to expected schema.
"""

import asyncio
from typing import List

from aikernel import LLMModelAlias, LLMRouter, LLMSystemMessage, LLMUserMessage
from pydantic import Field

from goose import Result, task


# Define a structured result type
class RecipeResult(Result):
    title: str = Field(description="The title of the recipe")
    ingredients: List[str] = Field(description="List of ingredients needed")
    steps: List[str] = Field(description="Step-by-step cooking instructions")
    prep_time_minutes: int = Field(description="Preparation time in minutes")
    cooking_time_minutes: int = Field(description="Cooking time in minutes")


@task
async def generate_recipe(*, ingredient: str) -> RecipeResult:
    """Generate a recipe that uses the specified ingredient."""
    router = LLMRouter[LLMModelAlias](
        model_list=[{"model_name": "gemini-2.0-flash", "litellm_params": {"model": "gemini/gemini-2.0-flash"}}],
        fallbacks=[],
    )
    
    system_message = LLMSystemMessage(
        content=f"""You are a helpful recipe generator.
        Please create a recipe that uses {ingredient} as a main ingredient.
        Return only the structured recipe information."""
    )
    
    user_message = LLMUserMessage(content=f"Create a recipe using {ingredient}")
    
    return await router.structured_generate(
        model="gemini-2.0-flash",
        messages=[system_message, user_message],
        response_model=RecipeResult,
    )


async def main():
    # Generate a recipe with structured output
    recipe = await generate_recipe(ingredient="avocado")
    
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


if __name__ == "__main__":
    asyncio.run(main())