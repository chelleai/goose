"""
Example demonstrating run persistence.

This example shows how Goose can save and restore flow runs,
allowing for resuming work or reviewing past executions.

NOTE: This is a demo file that illustrates the pattern without making actual LLM calls.
"""

import asyncio
import json
import os
import time
from typing import Dict, List, Optional

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
    """Structured note with title, content, and tags."""
    title: str = Field(description="Title of the note")
    content: str = Field(description="Content of the note")
    tags: List[str] = Field(description="Tags associated with the note")


class NoteTakingFlowArguments(FlowArguments):
    """Arguments for the note-taking flow."""
    topic: str
    context: Optional[str] = None


# Mock implementation for note generation
def mock_note_generator(topic: str, context: Optional[str] = None) -> TaskNote:
    """Generate a mock note based on the topic and context."""
    # Basic note for Machine Learning
    if "Machine Learning" in topic and "beginner" in (context or "").lower():
        return TaskNote(
            title="Introduction to Machine Learning",
            content="""Machine Learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.

The learning process begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers learn automatically without human intervention or assistance and adjust actions accordingly.

Key concepts for beginners:
1. Supervised Learning: Training with labeled data
2. Unsupervised Learning: Finding patterns in unlabeled data  
3. Reinforcement Learning: Learning through trial and error
4. Overfitting: When models perform well on training data but poorly on new data
5. Feature Engineering: Selecting relevant variables for your model

Popular algorithms include linear regression, decision trees, random forests, and neural networks.""",
            tags=["machine learning", "AI", "supervised learning", "algorithms", "beginner"]
        )
    # Advanced note that builds on the basic one
    elif "Advanced" in topic and "machine learning" in (context or "").lower():
        return TaskNote(
            title="Advanced Machine Learning Concepts",
            content="""Building on the fundamentals of machine learning, advanced concepts explore more sophisticated algorithms, techniques, and applications.

Deep Learning: Neural networks with multiple layers that can learn hierarchical representations. Frameworks like TensorFlow and PyTorch have made implementation more accessible.

Ensemble Methods: Combining multiple models to improve performance. Examples include bagging (Random Forests), boosting (XGBoost, LightGBM), and stacking.

Transfer Learning: Leveraging pre-trained models and adapting them to new tasks, significantly reducing data requirements and training time.

Generative Models: Models that can generate new content, such as GANs (Generative Adversarial Networks) for image synthesis or transformer-based models like GPT for text generation.

Reinforcement Learning: Advanced techniques like Deep Q-Networks (DQN), Proximal Policy Optimization (PPO), and AlphaGo's Monte Carlo Tree Search.

Explainable AI (XAI): Methods to understand and interpret model decisions, critical for applications in healthcare, finance, and legal domains.

AutoML: Automating the process of model selection, hyperparameter tuning, and feature engineering.

These advanced concepts are driving innovations in computer vision, natural language processing, robotics, and many other fields.""",
            tags=["deep learning", "transfer learning", "ensemble methods", "generative models", "XAI", "advanced techniques"]
        )
    # Generic note for any other topic
    else:
        return TaskNote(
            title=f"Notes on {topic}",
            content=f"This is a note about {topic}. " + (f"Additional context: {context}" if context else "No additional context provided."),
            tags=[topic.lower().replace(" ", "-"), "general-notes"]
        )


@task
async def create_note(*, agent: Agent, topic: str, context: Optional[str] = None) -> TaskNote:
    """Create a note on the specified topic.
    
    In a real implementation, this would call an LLM through the agent.
    For this example, we use a mock implementation.
    """
    print(f"Generating note on topic: {topic}" + (f" with context: {context}" if context else ""))
    
    # In a real implementation, you would use the agent to call the LLM:
    # context_str = f"\nContext: {context}" if context else ""
    # return await agent(
    #     messages=[...],
    #     model="your-model",
    #     task_name="create_note",
    #     response_model=TaskNote
    # )
    
    # For this example, we use a mock implementation
    return mock_note_generator(topic, context)


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
    print("\n--- Creating new flow run ---")
    
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
        print("\n--- New Note Created ---")
        print(f"Title: {note.title}")
        print(f"Tags: {', '.join(note.tags)}")
        print(f"Content Preview: {note.content[:150]}...")
    
    return run_id


async def resume_existing_flow(run_id: str):
    """Resume an existing flow run from persistence."""
    print(f"\n--- Resuming existing flow run: {run_id} ---")
    
    # Load the persisted flow run
    async with note_taking_flow.start_run(run_id=run_id) as run:
        # Retrieve the previously created note
        previous_note = run.get_result(task=create_note)
        print("\n--- Loaded Note From Persistence ---")
        print(f"Title: {previous_note.title}")
        print(f"Tags: {', '.join(previous_note.tags)}")
        print(f"Content Preview: {previous_note.content[:150]}...")
        
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
        print("\n--- Updated Note ---")
        print(f"Title: {updated_note.title}")
        print(f"Tags: {', '.join(updated_note.tags)}")
        print(f"Content Preview: {updated_note.content[:150]}...")


async def main():
    """Run the note-taking flow and demonstrate persistence."""
    print("=== Run Persistence Example ===")
    print("This example demonstrates how Goose can save and restore flow runs,")
    print("allowing for resuming work or reviewing past executions.\n")
    
    # Create storage directory
    os.makedirs("./saved_runs", exist_ok=True)
    
    # Create a new flow run and persist it
    run_id = await run_new_flow()
    
    # Simulate time passing between sessions
    print("\nSimulating time passing... (in a real application, this could be hours or days later)")
    time.sleep(2)
    
    # Resume the existing flow run from persistence
    await resume_existing_flow(run_id)
    
    print("\nNOTE: The flow state was saved to and loaded from disk using a custom")
    print("FlowRunStore implementation. This enables stateful, persistent workflows")
    print("that can be resumed across different sessions or machines.")


if __name__ == "__main__":
    asyncio.run(main())