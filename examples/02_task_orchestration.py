"""
Example demonstrating task orchestration.

This example shows how to create multiple tasks and orchestrate them
in a flow to create a multi-step workflow.
"""

import asyncio
from typing import List

from aikernel import LLMModelAlias, LLMRouter, LLMSystemMessage, LLMUserMessage
from pydantic import BaseModel, Field

from goose import Agent, FlowArguments, Result, TextResult, flow, task


class StoryTheme(Result):
    setting: str = Field(description="The setting of the story")
    characters: List[str] = Field(description="Main characters in the story")
    genre: str = Field(description="The genre of the story")


class StoryOutline(Result):
    title: str = Field(description="The title of the story")
    chapters: List[str] = Field(description="Brief outline of each chapter")
    main_conflict: str = Field(description="The main conflict in the story")


class StoryFlowArguments(FlowArguments):
    topic: str
    target_audience: str


# Task 1: Generate story theme
@task
async def generate_theme(*, agent: Agent, topic: str, audience: str) -> StoryTheme:
    """Generate a theme for the story based on topic and audience."""
    router = LLMRouter[LLMModelAlias](
        model_list=[{"model_name": "gemini-2.0-flash", "litellm_params": {"model": "gemini/gemini-2.0-flash"}}],
        fallbacks=[],
    )
    
    return await agent(
        messages=[
            LLMSystemMessage(content=f"You are a creative writing assistant helping generate story themes."),
            LLMUserMessage(content=f"Generate a story theme about '{topic}' for {audience}.")
        ],
        model="gemini-2.0-flash",
        task_name="generate_theme",
        response_model=StoryTheme,
        router=router
    )


# Task 2: Generate story outline based on theme
@task
async def generate_outline(*, agent: Agent, theme: StoryTheme) -> StoryOutline:
    """Generate a story outline based on the theme."""
    router = LLMRouter[LLMModelAlias](
        model_list=[{"model_name": "gemini-2.0-flash", "litellm_params": {"model": "gemini/gemini-2.0-flash"}}],
        fallbacks=[],
    )
    
    theme_description = (
        f"Setting: {theme.setting}\n"
        f"Characters: {', '.join(theme.characters)}\n"
        f"Genre: {theme.genre}"
    )
    
    return await agent(
        messages=[
            LLMSystemMessage(content="You are a creative writing assistant helping develop story outlines."),
            LLMUserMessage(content=f"Create a story outline based on this theme:\n\n{theme_description}")
        ],
        model="gemini-2.0-flash",
        task_name="generate_outline",
        response_model=StoryOutline,
        router=router
    )


# Task 3: Generate a first paragraph based on outline
@task
async def generate_opening(*, agent: Agent, outline: StoryOutline) -> TextResult:
    """Generate an opening paragraph based on the outline."""
    router = LLMRouter[LLMModelAlias](
        model_list=[{"model_name": "gemini-2.0-flash", "litellm_params": {"model": "gemini/gemini-2.0-flash"}}],
        fallbacks=[],
    )
    
    outline_description = (
        f"Title: {outline.title}\n"
        f"Main Conflict: {outline.main_conflict}\n"
        f"First Chapter Theme: {outline.chapters[0] if outline.chapters else 'Introduction'}"
    )
    
    return await agent(
        messages=[
            LLMSystemMessage(content="You are a creative writing assistant."),
            LLMUserMessage(content=f"Write an engaging opening paragraph for a story with this outline:\n\n{outline_description}")
        ],
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
    # Run the story generation flow
    async with story_generation_flow.start_run(run_id="story-generation-1") as run:
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
        
        print("STORY GENERATION RESULTS\n")
        
        print("Theme:")
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


if __name__ == "__main__":
    asyncio.run(main())