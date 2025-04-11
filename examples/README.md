# Goose Examples

This directory contains examples that demonstrate the key features of the Goose framework.

## Example Files

1. **Structured LLM Interactions** (`01_structured_responses.py`)  
   Demonstrates how to create structured response types and ensure that LLM outputs conform to the expected schema.

2. **Task Orchestration** (`02_task_orchestration.py`)  
   Shows how to create multiple tasks and orchestrate them in a flow to create a multi-step workflow.

3. **Stateful Conversations** (`03_stateful_conversations.py`)  
   Demonstrates how to maintain conversation history across multiple interactions with a task.

4. **Result Caching** (`04_result_caching.py`)  
   Showcases how Goose automatically caches results based on input hashing and only regenerates results when inputs change.

5. **Iterative Refinement** (`05_iterative_refinement.py`)  
   Shows how to refine a structured result by providing feedback and using the refine method to improve it.

6. **Result Validation** (`06_result_validation.py`)  
   Demonstrates how Goose validates model outputs against expected schemas and handles validation errors.

7. **Run Persistence** (`07_run_persistence.py`)  
   Shows how Goose can save and restore flow runs, allowing for resuming work or reviewing past executions.

8. **Custom Logging** (`08_custom_logging.py`)  
   Demonstrates how to implement custom loggers to track metrics about LLM interactions.

## Running the Examples

Each example is a standalone Python file that can be run directly:

```bash
python examples/01_structured_responses.py
```

Note that these examples require the following dependencies:
- Python 3.12+
- Goose framework
- aikernel
- pydantic

The examples use the Gemini 2.0 Flash model, so you'll need appropriate API credentials configured.