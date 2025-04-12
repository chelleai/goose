"""
Example demonstrating stateful conversations.

This example shows how to maintain conversation history and context
across multiple interactions with a task.

NOTE: This is a demo file that illustrates the pattern without making actual LLM calls.
"""

import asyncio
import os

from goose import Agent, FlowArguments, TextResult, flow, task


class TutorFlowArguments(FlowArguments):
    """Arguments for the tutoring flow."""
    subject: str
    student_level: str


# Mock implementations for conversations
def mock_initial_explanation() -> str:
    """Return a mock explanation of quantum entanglement."""
    return """Quantum entanglement is a phenomenon in quantum physics where two or more particles become correlated in such a way that the quantum state of each particle cannot be described independently of the others. This means that when you measure one particle, you instantly know information about the other entangled particle(s), regardless of the distance between them.

Einstein famously referred to this as "spooky action at a distance" because it seems to violate the principle that information cannot travel faster than the speed of light. However, it's important to understand that while the correlation is instantaneous, it doesn't allow for faster-than-light communication of useful information.

At a technical level, entanglement occurs when particles interact physically and then become separated while maintaining their quantum connection. The mathematical formulation involves non-separable quantum states, meaning the quantum state of the entire system cannot be factored into the states of the individual particles."""


def mock_followup_responses(question_number: int, question: str) -> str:
    """Return mock responses to follow-up questions."""
    responses = {
        1: """Let me explain quantum entanglement more simply:

Imagine you have two coins in a special box, and they're magically connected. When you take them out and give one to your friend who travels far away, something strange happens. If you flip your coin and it lands on heads, your friend's coin will ALWAYS land on tails when they flip it - instantly! This happens even though the coins aren't physically connected anymore.

The weird part is that the coins don't "decide" to be heads or tails until you actually look at them. And somehow, when one coin "decides," the other one immediately "knows" what to do, no matter how far apart they are. 

Scientists are puzzled because it seems like the coins are communicating faster than the speed of light, which shouldn't be possible according to physics. But they're not actually sending information - they're just mysteriously connected in a way we can't explain with our everyday understanding of the world.""",
        
        2: """Quantum entanglement is absolutely fundamental to quantum computing!

In classical computers, bits are either 0 or 1. But quantum computers use quantum bits or "qubits" that can exist in multiple states simultaneously thanks to superposition. Entanglement takes this further by linking qubits together, creating a system where the number of states grows exponentially with each added qubit.

When qubits are entangled, operations on one can affect others instantly, allowing quantum computers to process vast amounts of information in parallel. This gives quantum computers their theoretical advantage for certain problems.

Specific applications include:

1. Shor's algorithm: Uses entanglement to factor large numbers exponentially faster than classical computers, potentially breaking common encryption methods.

2. Grover's search algorithm: Uses entanglement to find items in an unsorted database quadratically faster than classical approaches.

3. Quantum simulation: Directly uses entanglement to model quantum systems that are impossible to simulate efficiently on classical computers.

Without entanglement, quantum computers would lose much of their computational advantage.""",
        
        3: """Quantum entanglement has several important real-world applications:

1. Quantum cryptography and secure communications: Quantum key distribution (QKD) uses entangled particles to create encryption keys that are theoretically unhackable. Any attempt to intercept the key would break the entanglement and reveal the intrusion.

2. Quantum teleportation: Not Star Trek-style teleportation, but a technique to transfer quantum states between particles. This is crucial for quantum networks and eventually a quantum internet.

3. Quantum sensors: Entangled particles can create extremely sensitive measurement devices. For example, improved atomic clocks, gravitational wave detectors, and medical imaging.

4. Quantum computing: As previously mentioned, entanglement is essential for quantum algorithms that solve certain problems exponentially faster than classical computers.

5. Fundamental physics research: Studying entanglement helps physicists understand the nature of reality, space-time, and the foundations of quantum mechanics.

Companies like IBM, Google, and startups like QuintessenceLabs are already commercializing some of these applications, particularly in quantum computing and quantum security."""
    }
    
    return responses.get(question_number, f"Response to: {question}")


@task
async def explain_concept(*, agent: Agent, concept: str) -> TextResult:
    """Explain a concept and maintain conversation history.
    
    In a real implementation, this would call an LLM through the agent.
    For this example, we use a mock implementation.
    """
    print(f"Generating explanation for concept: {concept}")
    
    # In a real implementation, you would use the agent to call the LLM:
    # return await agent(
    #     messages=[...],
    #     model="your-model",
    #     task_name="explain_concept",
    #     response_model=TextResult
    # )
    
    # For this example, we use a mock implementation
    return TextResult(text=mock_initial_explanation())


@flow
async def tutoring_flow(*, flow_arguments: TutorFlowArguments, agent: Agent) -> None:
    """Flow for a tutoring session that maintains conversation history."""
    # Initial explanation
    await explain_concept(agent=agent, concept="quantum entanglement")


async def main():
    """Run the tutoring flow and demonstrate stateful conversations."""
    # Create a unique run ID
    run_id = f"tutoring-{os.getpid()}"
    
    print("=== Stateful Conversations Example ===")
    print("This example demonstrates how to maintain conversation history")
    print("across multiple interactions with a task.\n")
    
    # Start a flow run for the tutoring session
    async with tutoring_flow.start_run(run_id=run_id) as run:
        # Initialize the flow
        await tutoring_flow.generate(
            TutorFlowArguments(
                subject="Physics",
                student_level="Undergraduate"
            )
        )
        
        # Get the initial explanation
        initial_explanation = run.get_result(task=explain_concept)
        print("\n--- Initial Explanation ---")
        print(initial_explanation.text)
        print("\n" + "-" * 50)
        
        # Ask follow-up questions using the same conversation context
        follow_up_questions = [
            "Can you explain it in simpler terms?",
            "How is this related to quantum computing?",
            "What are some real-world applications?"
        ]
        
        # In a real implementation, these would be actual calls:
        # from aikernel import LLMUserMessage, LLMMessagePart
        # response = await explain_concept.ask(
        #     user_message=LLMUserMessage(parts=[LLMMessagePart(content=question)]),
        #     model="your-model",
        #     router=router
        # )
        
        # For this example, we'll simulate the ask method's behavior
        for i, question in enumerate(follow_up_questions, 1):
            print(f"\n--- Follow-up Question {i}: {question} ---")
            
            # In a real implementation, this would actually call the model
            # and maintain the conversation history
            mock_response = mock_followup_responses(i, question)
            print(mock_response)
            
            # In the real task.ask() method, this would happen internally:
            # 1. The question would be added to the conversation history
            # 2. The full history would be sent to the LLM
            # 3. The response would be added back to the conversation history
            print("\nNOTE: The system has automatically maintained the conversation context.")
            print("The LLM would receive the full history of the conversation,")
            print("allowing it to provide coherent follow-up responses.")
            
            print("-" * 50)
        
        print("\nThis example demonstrates how Goose maintains conversation state")
        print("across multiple interactions, allowing for natural follow-up questions.")


if __name__ == "__main__":
    asyncio.run(main())