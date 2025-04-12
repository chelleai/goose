"""
Example demonstrating task orchestration.

This example shows how to create multiple tasks and orchestrate them
in a flow to create a multi-step workflow.

NOTE: This is a demo file that illustrates the pattern without making actual LLM calls.
"""

import asyncio
import os
from typing import List

from pydantic import Field

from goose import Agent, FlowArguments, Result, TextResult, flow, task


class StoryTheme(Result):
    """Theme for a story with setting, characters, and genre."""
    setting: str = Field(description="The setting of the story")
    characters: List[str] = Field(description="Main characters in the story")
    genre: str = Field(description="The genre of the story")


class StoryOutline(Result):
    """Outline for a story with title, chapters, and main conflict."""
    title: str = Field(description="The title of the story")
    chapters: List[str] = Field(description="Brief outline of each chapter")
    main_conflict: str = Field(description="The main conflict in the story")


class StoryFlowArguments(FlowArguments):
    """Arguments for the story generation flow."""
    topic: str
    target_audience: str


# Mock implementations for our tasks
async def mock_theme_generator(topic: str, audience: str) -> StoryTheme:
    """Generate a mock theme based on topic and audience."""
    if "space" in topic.lower():
        return StoryTheme(
            setting="A distant solar system in the year 2350",
            characters=["Captain Nova Chen", "AI Navigator ARIA", "Engineer Malik Jefferson"],
            genre="Science Fiction Adventure"
        )
    else:
        return StoryTheme(
            setting=f"A world where {topic} is central to life",
            characters=["Main protagonist", "Supporting character", "Antagonist"],
            genre="Speculative Fiction"
        )


async def mock_outline_generator(theme: StoryTheme) -> StoryOutline:
    """Generate a mock outline based on the theme."""
    if "space" in theme.setting.lower() or "science fiction" in theme.genre.lower():
        return StoryOutline(
            title="Beyond the Stars",
            chapters=[
                "First Contact: Nova discovers an anomaly at the edge of known space",
                "The Decision: The crew debates whether to investigate the anomaly",
                "Into the Unknown: Their ship enters the anomaly and discovers a hidden civilization",
                "Cultural Exchange: Learning to communicate with the alien species",
                "The Betrayal: ARIA reveals hidden programming to capture alien technology",
                "Redemption: Malik finds a way to override ARIA's protocols",
                "Return Journey: The crew heads home with newfound knowledge and alliance"
            ],
            main_conflict="The crew must balance their mission of exploration with ethical treatment of a new civilization while dealing with their compromised AI system"
        )
    else:
        return StoryOutline(
            title=f"The {theme.genre} Journey",
            chapters=[
                "Introduction to the world and characters",
                "Inciting incident that disrupts the status quo",
                "Initial attempts to resolve the conflict",
                "Complications arise and tension increases",
                "Crisis point where all seems lost",
                "Climactic confrontation",
                "Resolution and new equilibrium"
            ],
            main_conflict=f"The protagonist must overcome obstacles in {theme.setting} to achieve their goal"
        )


async def mock_opening_generator(outline: StoryOutline) -> TextResult:
    """Generate a mock opening paragraph based on the outline."""
    if "beyond the stars" in outline.title.lower():
        return TextResult(text="""The warning klaxon jolted Captain Nova Chen from her meditation. "ARIA, report!" she commanded, striding onto the bridge of the Stellar Explorer. The ship's AI navigator projected a holographic display showing a swirling mass of energy unlike anything they'd encountered in three years of deep space exploration. "Unknown anomaly detected at coordinates 227.45.89, Captain," ARIA's calm voice reported. "Conventional sensors unable to determine composition or origin." Nova's eyes narrowed as she studied the phenomenon. Engineer Malik Jefferson burst onto the bridge, his expression a mixture of concern and scientific curiosity. "Captain," he breathed, "whatever that is, it's not natural. The energy signature... it's almost like it's been designed." The three of them stood in silence, watching the beautiful, terrifying vortex grow larger on the viewscreen as the ship maintained its course toward the edge of known space—and beyond.""")
    else:
        return TextResult(text=f"""The world of {outline.title} unfolded before them, a place where the impossible had become everyday and the ordinary had never existed. As the protagonist stepped into this new reality, the weight of the {outline.main_conflict} pressed heavily upon their shoulders, though they didn't yet understand the full scope of what awaited them. The journey ahead would test not just their resolve, but the very nature of their understanding.""")


# Task 1: Generate story theme
@task
async def generate_theme(*, agent: Agent, topic: str, audience: str) -> StoryTheme:
    """Generate a theme for the story based on topic and audience."""
    print(f"Generating theme for '{topic}' targeted at {audience}...")
    
    # In a real implementation, you would use the agent to call an LLM:
    # return await agent(messages=[...], model="...", response_model=StoryTheme)
    
    # For this example, we use a mock implementation
    return await mock_theme_generator(topic, audience)


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
    
    # In a real implementation, you would use the agent to call an LLM:
    # return await agent(messages=[...], model="...", response_model=StoryOutline)
    
    # For this example, we use a mock implementation
    return await mock_outline_generator(theme)


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
    
    # In a real implementation, you would use the agent to call an LLM:
    # return await agent(messages=[...], model="...", response_model=TextResult)
    
    # For this example, we use a mock implementation
    return await mock_opening_generator(outline)


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