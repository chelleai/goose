# Goose

Goose is a framework for building LLM-based agents and workflows with strong typing and state management. Here's what's fundamentally possible:

1. Structured LLM interactions - Organize model calls with typed inputs/outputs
2. Task orchestration - Create reusable tasks that can be composed into flows
3. Stateful conversations - Maintain conversation history and model outputs
4. Result caching - Avoid redundant computation based on input hashing
5. Iterative refinement - Enhance results through progressive feedback loops
6. Result validation - Ensure model outputs conform to expected schemas
7. Run persistence - Save and reload workflow executions
8. Custom logging - Track telemetry and performance metrics

It enables building reliable, maintainable AI applications with proper error handling, state tracking, and flow control while ensuring type safety throughout.

## Key Features

### Structured LLM Interactions

Organize model calls with typed inputs and outputs using Pydantic models. This ensures that responses from language models conform to expected structures.

```mermaid
graph LR
    A[User Input] --> B[Agent]
    B --> C[LLM Model]
    C --> D[Structured Response]
    D --> E[Validated Result]
    E --> F[Application Logic]
    
    classDef user fill:#f9f,stroke:#333,stroke-width:2px
    classDef llm fill:#bbf,stroke:#333,stroke-width:2px
    classDef validation fill:#bfb,stroke:#333,stroke-width:2px
    
    class A user
    class C llm
    class D,E validation
```

### Task Orchestration

Create reusable tasks that can be composed into flows. Tasks are decorated functions that handle specific operations, while flows coordinate multiple tasks.

```mermaid
graph TD
    A[Flow] --> B[Task 1]
    A --> C[Task 2]
    A --> D[Task 3]
    B --> E[Result 1]
    C --> F[Result 2]
    D --> G[Result 3]
    E --> H[Flow Output]
    F --> H
    G --> H
    
    classDef flow fill:#f9f,stroke:#333,stroke-width:2px
    classDef task fill:#bbf,stroke:#333,stroke-width:2px
    classDef result fill:#bfb,stroke:#333,stroke-width:2px
    
    class A flow
    class B,C,D task
    class E,F,G,H result
```

### Stateful Conversations

Maintain conversation history and model outputs across multiple interactions. The framework tracks the state of each task in a flow.

```mermaid
sequenceDiagram
    participant User
    participant Flow
    participant Task
    participant Agent
    participant LLM
    
    User->>Flow: Start Conversation
    Flow->>Task: Execute
    Task->>Agent: Generate Response
    Agent->>LLM: Send Messages
    LLM-->>Agent: Generate Response
    Agent-->>Task: Store Result
    Task-->>Flow: Update State
    Flow-->>User: Return Result
    
    User->>Flow: Follow-up Question
    Flow->>Task: Get State
    Task->>Agent: Send Previous Context + New Question
    Agent->>LLM: Send Updated Messages
    LLM-->>Agent: Generate Response
    Agent-->>Task: Update Conversation
    Task-->>Flow: Update State
    Flow-->>User: Return Result
```

### Result Caching

Avoid redundant computation by caching results based on input hashing. The framework automatically detects when inputs change and only regenerates results when necessary.

```mermaid
flowchart TD
    A[Task Call] --> B{Inputs Changed?}
    B -- Yes --> C[Execute Task]
    B -- No --> D[Return Cached Result]
    C --> E[Cache Result]
    E --> F[Return Result]
    D --> F
    
    classDef decision fill:#f9f,stroke:#333,stroke-width:2px
    classDef action fill:#bbf,stroke:#333,stroke-width:2px
    classDef cache fill:#bfb,stroke:#333,stroke-width:2px
    
    class B decision
    class A,C,F action
    class D,E cache
```

### Iterative Refinement

Enhance results through progressive feedback loops. The framework supports asking follow-up questions about results and refining them based on feedback.

```mermaid
sequenceDiagram
    participant User
    participant Task
    participant Agent
    participant LLM
    
    User->>Task: Generate Initial Result
    Task->>Agent: Send Request
    Agent->>LLM: Generate Structured Output
    LLM-->>Agent: Return Output
    Agent-->>Task: Store Result
    Task-->>User: Return Result
    
    User->>Task: Request Refinement
    Task->>Agent: Send Feedback + Original Result
    Agent->>LLM: Generate Find/Replace Operations
    LLM-->>Agent: Return Changes
    Agent-->>Task: Apply Changes to Result
    Task-->>User: Return Refined Result
```

### Result Validation

Ensure model outputs conform to expected schemas using Pydantic validation. All results must conform to predefined models.

```mermaid
flowchart LR
    A[LLM Response] --> B[Parse JSON]
    B --> C{Valid Schema?}
    C -- Yes --> D[Return Validated Result]
    C -- No --> E[Raise Error]
    
    classDef input fill:#bbf,stroke:#333,stroke-width:2px
    classDef validation fill:#f9f,stroke:#333,stroke-width:2px
    classDef output fill:#bfb,stroke:#333,stroke-width:2px
    classDef error fill:#fbb,stroke:#333,stroke-width:2px
    
    class A input
    class B,C validation
    class D output
    class E error
```

### Run Persistence

Save and reload workflow executions. The framework provides interfaces for storing flow runs, allowing for resuming work or reviewing past executions.

```mermaid
graph TD
    A[Start Flow] --> B[Create Flow Run]
    B --> C[Execute Tasks]
    C --> D[Save Run State]
    D --> E[End Flow]
    
    F[Later Time] --> G[Load Saved Run]
    G --> H[Resume Execution]
    H --> D
    
    classDef flow fill:#f9f,stroke:#333,stroke-width:2px
    classDef execution fill:#bbf,stroke:#333,stroke-width:2px
    classDef storage fill:#bfb,stroke:#333,stroke-width:2px
    
    class A,E,F flow
    class B,C,H execution
    class D,G storage
```

### Custom Logging

Track telemetry and performance metrics. The framework supports custom loggers to record model usage, token counts, and execution time.

```mermaid
flowchart TD
    A[Agent Call] --> B[Execute LLM Request]
    B --> C[Record Metrics]
    C --> D{Custom Logger?}
    D -- Yes --> E[Send to Custom Logger]
    D -- No --> F[Log to Default Logger]
    E --> G[Return Result]
    F --> G
    
    classDef action fill:#bbf,stroke:#333,stroke-width:2px
    classDef logging fill:#bfb,stroke:#333,stroke-width:2px
    classDef decision fill:#f9f,stroke:#333,stroke-width:2px
    
    class A,B,G action
    class C,E,F logging
    class D decision
```

## Building with Goose

Goose enables building reliable, maintainable AI applications with proper error handling, state tracking, and flow control while ensuring type safety throughout. This approach reduces common issues in LLM applications like:

- Type inconsistencies in model responses
- Loss of context between interactions
- Redundant LLM calls for identical inputs
- Difficulty in resuming interrupted workflows
- Lack of structured error handling

Start building more robust LLM applications with Goose's typed, stateful approach to agent development.

## Installation and Package Management

Goose uses `uv` for package management. Never use pip with this project.

```bash
# Install dependencies
uv add <package-name>

# Update dependencies file
uv sync

# Run commands
uv run <command>
```
