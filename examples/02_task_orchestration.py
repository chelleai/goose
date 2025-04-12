"""
Example demonstrating task orchestration.

This example shows how to create multiple tasks and orchestrate them
in a flow to create a multi-step workflow.
"""

import asyncio
import os

from aikernel import LLMMessagePart, LLMSystemMessage, LLMUserMessage, get_router
from pydantic import Field

from goose import Agent, FlowArguments, Result, TextResult, flow, task


class StoryTheme(Result):
    """Theme for a story with setting, characters, and genre."""
    setting: str = Field(description="The setting of the story")
    characters: list[str] = Field(description="Main characters in the story")
    genre: str = Field(description="The genre of the story")


class StoryOutline(Result):
    """Outline for a story with title, chapters, and main conflict."""
    title: str = Field(description="The title of the story")
    chapters: list[str] = Field(description="Brief outline of each chapter")
    main_conflict: str = Field(description="The main conflict in the story")


class StoryFlowArguments(FlowArguments):
    """Arguments for the story generation flow."""
    topic: str
    target_audience: str




# Task 1: Generate story theme
@task
async def generate_theme(*, agent: Agent, topic: str, audience: str) -> StoryTheme:
    """Generate a theme for the story based on topic and audience."""
    print(f"Generating theme for '{topic}' targeted at {audience}...")
    
    # Create a router for Gemini 2.0 Flash
    router = get_router(models=("gemini-2.0-flash",))
    
    # System message with instructions
    system_message = LLMSystemMessage(
        parts=[LLMMessagePart(content=f"You are a creative writing assistant. Create a story theme about {topic} for {audience}.")]
    )
    
    # User request message
    user_message = LLMUserMessage(
        parts=[LLMMessagePart(content=f"Please create a story theme about {topic} for {audience}. Include a setting, main characters, and genre.")]
    )
    
    # Make the actual LLM call
    return await agent(
        messages=[system_message, user_message],
        model="gemini-2.0-flash",
        task_name="generate_theme",
        response_model=StoryTheme,
        router=router
    )


# Task 2: Generate story outline based on theme
@task
async def generate_outline(*, agent: Agent, theme: StoryTheme) -> StoryOutline:
    """Generate a story outline based on the theme."""
    theme_description = (
        f"Setting: {theme.setting}\n"
        f"Characters: {', '.join(theme.characters)}\n"
        f"Genre: {theme.genre}"
    )
    print(f"Generating outline based on theme:\n{theme_description}")
    
    # Create a router for Gemini 2.0 Flash
    router = get_router(models=("gemini-2.0-flash",))
    
    # System message with instructions
    system_message = LLMSystemMessage(
        parts=[LLMMessagePart(content="You are a creative writing assistant. Create a story outline based on the provided theme.")]
    )
    
    # User request message
    user_message = LLMUserMessage(
        parts=[LLMMessagePart(content=f"Please create a story outline based on this theme:\n{theme_description}\n\nInclude a title, a list of chapters, and the main conflict.")]
    )
    
    # Make the actual LLM call
    return await agent(
        messages=[system_message, user_message],
        model="gemini-2.0-flash",
        task_name="generate_outline",
        response_model=StoryOutline,
        router=router
    )


# Task 3: Generate a first paragraph based on outline
@task
async def generate_opening(*, agent: Agent, outline: StoryOutline) -> TextResult:
    """Generate an opening paragraph based on the outline."""
    outline_description = (
        f"Title: {outline.title}\n"
        f"Main Conflict: {outline.main_conflict}\n"
        f"First Chapter: {outline.chapters[0] if outline.chapters else 'Introduction'}"
    )
    print(f"Generating opening paragraph based on outline:\n{outline_description}")
    
    # Create a router for Gemini 2.0 Flash
    router = get_router(models=("gemini-2.0-flash",))
    
    # System message with instructions
    system_message = LLMSystemMessage(
        parts=[LLMMessagePart(content="You are a creative writing assistant. Write an engaging opening paragraph for a story based on the provided outline.")]
    )
    
    # User request message
    user_message = LLMUserMessage(
        parts=[LLMMessagePart(content=f"Please write an engaging opening paragraph for a story with this outline:\n{outline_description}")]
    )
    
    # Make the actual LLM call
    return await agent(
        messages=[system_message, user_message],
        model="gemini-2.0-flash",
        task_name="generate_opening",
        response_model=TextResult,
        router=router
    )


# Define a flow that connects these tasks
@flow
async def story_generation_flow(*, flow_arguments: StoryFlowArguments, agent: Agent) -> None:
    """Flow that orchestrates the story generation process."""
    # Generate the story theme
    theme = await generate_theme(
        agent=agent,
        topic=flow_arguments.topic,
        audience=flow_arguments.target_audience
    )
    
    # Use the theme to generate an outline
    outline = await generate_outline(
        agent=agent,
        theme=theme
    )
    
    # Use the outline to generate an opening paragraph
    opening = await generate_opening(
        agent=agent,
        outline=outline
    )


async def main():
    """Run the story generation flow and display the results."""
    # Create a unique run ID
    run_id = f"story-{os.getpid()}"
    
    print("=== Task Orchestration Example ===")
    print("This example demonstrates how to create multiple tasks")
    print("and orchestrate them in a flow to create a multi-step workflow.\n")
    
    # Run the story generation flow
    async with story_generation_flow.start_run(run_id=run_id) as run:
        await story_generation_flow.generate(
            StoryFlowArguments(
                topic="space exploration",
                target_audience="young adults"
            )
        )
        
        # Retrieve and display the results
        theme = run.get_result(task=generate_theme)
        outline = run.get_result(task=generate_outline)
        opening = run.get_result(task=generate_opening)
        
        print("\n--- Story Generation Results ---")
        
        print("\nTheme:")
        print(f"Setting: {theme.setting}")
        print(f"Characters: {', '.join(theme.characters)}")
        print(f"Genre: {theme.genre}")
        
        print("\nOutline:")
        print(f"Title: {outline.title}")
        print(f"Main Conflict: {outline.main_conflict}")
        print("Chapters:")
        for i, chapter in enumerate(outline.chapters, 1):
            print(f"  {i}. {chapter}")
        
        print("\nOpening Paragraph:")
        print(opening.text)
        
        print("\nNote: This example demonstrates how tasks can be composed in a flow,")
        print("with each task using the results of previous tasks as input.")


if __name__ == "__main__":
    asyncio.run(main())
