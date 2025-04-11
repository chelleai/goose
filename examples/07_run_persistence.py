"""
Example demonstrating run persistence.

This example shows how Goose can save and restore flow runs,
allowing for resuming work or reviewing past executions.
"""

import asyncio
import json
import os
import time
from typing import Dict, List, Optional

from aikernel import LLMModelAlias, LLMRouter, LLMSystemMessage, LLMUserMessage
from pydantic import Field

from goose import Agent, FlowArguments, Result, flow, task
from goose._internal.store import IFlowRunStore, SerializedFlowRun


# Custom flow run store implementation that saves to disk
class FileFlowRunStore(IFlowRunStore):
    """Flow run store that persists runs to JSON files on disk."""
    
    def __init__(self, storage_dir: str, flow_name: str):
        self.storage_dir = storage_dir
        self.flow_name = flow_name
        os.makedirs(storage_dir, exist_ok=True)
    
    def _get_file_path(self, run_id: str) -> str:
        """Get the file path for a run ID."""
        return os.path.join(self.storage_dir, f"{self.flow_name}_{run_id}.json")
    
    async def save(self, *, run_id: str, run: SerializedFlowRun) -> None:
        """Save a flow run to disk."""
        file_path = self._get_file_path(run_id)
        with open(file_path, 'w') as f:
            json.dump(run, f, indent=2)
        print(f"Run saved to {file_path}")
    
    async def get(self, *, run_id: str) -> Optional[SerializedFlowRun]:
        """Load a flow run from disk."""
        file_path = self._get_file_path(run_id)
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, 'r') as f:
            run_data = json.load(f)
        print(f"Run loaded from {file_path}")
        return run_data


class TaskNote(Result):
    title: str = Field(description="Title of the note")
    content: str = Field(description="Content of the note")
    tags: List[str] = Field(description="Tags associated with the note")


class NoteTakingFlowArguments(FlowArguments):
    topic: str
    context: Optional[str] = None


@task
async def create_note(*, agent: Agent, topic: str, context: Optional[str] = None) -> TaskNote:
    """Create a note on the specified topic."""
    router = LLMRouter[LLMModelAlias](
        model_list=[{"model_name": "gemini-2.0-flash", "litellm_params": {"model": "gemini/gemini-2.0-flash"}}],
        fallbacks=[],
    )
    
    context_str = f"\nContext: {context}" if context else ""
    
    return await agent(
        messages=[
            LLMSystemMessage(content="You are a helpful note-taking assistant."),
            LLMUserMessage(content=f"Create a structured note about: {topic}{context_str}")
        ],
        model="gemini-2.0-flash",
        task_name="create_note",
        response_model=TaskNote,
        router=router
    )


# Define a flow with custom persistence store
@flow(store=FileFlowRunStore(storage_dir="./saved_runs", flow_name="notes"))
async def note_taking_flow(*, flow_arguments: NoteTakingFlowArguments, agent: Agent) -> None:
    """Flow for creating and managing notes with persistence."""
    await create_note(
        agent=agent,
        topic=flow_arguments.topic,
        context=flow_arguments.context
    )


async def run_new_flow():
    """Create a new flow run and persist it."""
    print("Creating new flow run...")
    
    # Unique run ID for demonstration
    run_id = f"note-{int(time.time())}"
    
    # Start a new flow run
    async with note_taking_flow.start_run(run_id=run_id) as run:
        # Generate a note
        await note_taking_flow.generate(
            NoteTakingFlowArguments(
                topic="Machine Learning Fundamentals",
                context="For a beginner audience"
            )
        )
        
        # Display the result
        note = run.get_result(task=create_note)
        print("\nNEW NOTE CREATED:")
        print(f"Title: {note.title}")
        print(f"Tags: {', '.join(note.tags)}")
        print(f"Content: {note.content}")
    
    return run_id


async def resume_existing_flow(run_id: str):
    """Resume an existing flow run from persistence."""
    print(f"\nResuming existing flow run {run_id}...")
    
    # Load the persisted flow run
    async with note_taking_flow.start_run(run_id=run_id) as run:
        # Retrieve the previously created note
        previous_note = run.get_result(task=create_note)
        print("\nLOADED NOTE FROM PERSISTENCE:")
        print(f"Title: {previous_note.title}")
        print(f"Tags: {', '.join(previous_note.tags)}")
        print(f"Content: {previous_note.content[:100]}...")
        
        # Extract a tag to use as a refinement topic
        refinement_topic = previous_note.tags[0] if previous_note.tags else "advanced concepts"
        
        # Update the flow with more context for the note
        await note_taking_flow.generate(
            NoteTakingFlowArguments(
                topic=f"Advanced {previous_note.title}",
                context=f"Expanding on {refinement_topic}"
            )
        )
        
        # Show the updated note
        updated_note = run.get_result(task=create_note)
        print("\nUPDATED NOTE:")
        print(f"Title: {updated_note.title}")
        print(f"Tags: {', '.join(updated_note.tags)}")
        print(f"Content: {updated_note.content}")


async def main():
    # Create storage directory
    os.makedirs("./saved_runs", exist_ok=True)
    
    # Create a new flow run and persist it
    run_id = await run_new_flow()
    
    # Simulate time passing between sessions
    print("\nSimulating time passing... (in a real application, this could be hours or days later)")
    time.sleep(2)
    
    # Resume the existing flow run from persistence
    await resume_existing_flow(run_id)


if __name__ == "__main__":
    asyncio.run(main())