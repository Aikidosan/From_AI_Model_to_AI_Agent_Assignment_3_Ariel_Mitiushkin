# Requirements Document

## Introduction

The Customer Service Data Analyst Agent is an interactive AI-powered system that answers user questions about the Bitext Customer Service Tagged Training Dataset. The agent handles structured queries (e.g., "How many refund requests?"), unstructured/open-ended queries (e.g., "Summarize the FEEDBACK category"), and gracefully declines out-of-scope questions. It is built on a LangGraph ReAct graph with persistent memory, exposes tools via a FastMCP server, and uses Nebius Token Factory models exclusively for all LLM calls. Users interact through a Python CLI and optionally a Streamlit web UI.

## Glossary

- **Agent**: The LangGraph ReAct graph that orchestrates tool calls and produces answers.
- **Dataset**: The Bitext Customer Service Tagged Training Dataset (CSV/parquet file loaded at startup).
- **Category**: A top-level grouping of customer service intents in the Dataset (e.g., FEEDBACK, ORDER, REFUND).
- **Intent**: A specific customer intent within a Category (e.g., "track_refund", "cancel_order").
- **Query_Router**: The dedicated LangGraph node that classifies each incoming user query as structured, unstructured, or out-of-scope before routing to the appropriate processing path.
- **Tool**: A callable function exposed to the Agent and via the MCP_Server, with a clear name, description, and Pydantic input schema.
- **MCP_Server**: The FastMCP server that exposes Agent tools as MCP-compatible endpoints.
- **Session**: A named conversation context identified by a session ID, persisted via SqliteSaver or PostgresSaver.
- **User_Profile**: A per-user persistent record capturing name, frequent topics, and preferences, stored across restarts.
- **CLI**: The command-line interface entry point (`python main.py`).
- **Streamlit_UI**: The optional web-based chat interface built with Streamlit.
- **Query_Recommender**: The optional subsystem that suggests follow-up queries based on conversation history and User_Profile.
- **Nebius_LLM**: Any language model accessed exclusively through the Nebius Token Factory API.
- **Checkpointer**: The LangGraph persistence backend (SqliteSaver or PostgresSaver) used to store conversation state.

---

## Requirements

### Requirement 1: Dataset Loading and Access

**User Story:** As a data analyst, I want the agent to load and index the Bitext Customer Service dataset at startup, so that all queries operate on consistent, in-memory data without repeated file I/O.

#### Acceptance Criteria

1. WHEN the Agent starts, THE Agent SHALL load the Bitext Customer Service Tagged Training Dataset from a configurable file path into memory.
2. WHEN the Agent attempts to load the dataset during startup and the file is not found at the configured path, THE Agent SHALL emit a descriptive error message and exit with a non-zero status code.
3. WHEN the dataset is loaded, THE Agent SHALL validate that the expected columns (utterance, category, intent, and any tag columns) are present.
4. IF required columns are missing from the dataset, THEN THE Agent SHALL emit a descriptive error listing the missing columns and exit with a non-zero status code.
5. THE Agent SHALL make the loaded dataset available to all Tools without reloading it on each tool call.

---

### Requirement 2: Query Router

**User Story:** As a user, I want my queries classified before processing, so that structured queries get precise data answers, unstructured queries get narrative summaries, and out-of-scope queries are declined without hallucination.

#### Acceptance Criteria

1. WHEN a user submits a query, THE Query_Router SHALL classify it as exactly one of: `structured`, `unstructured`, or `out_of_scope` before any tool is invoked.
2. WHEN the Query_Router classifies a query as `out_of_scope`, THE Agent SHALL respond with a polite decline message that does not use Nebius_LLM general knowledge to answer the query.
3. WHEN the Query_Router classifies a query as `structured`, THE Agent SHALL route it to the ReAct tool-calling path.
4. WHEN the Query_Router classifies a query as `unstructured`, THE Agent SHALL route it to the summarization path using the summarize_category Tool, and SHALL allow that path to invoke additional structured Tools when needed to ground the summary in dataset facts.
5. THE Query_Router SHALL use a Nebius_LLM call with a focused classification prompt and SHALL NOT invoke dataset Tools during classification.

---

### Requirement 3: Core Dataset Tools

**User Story:** As a data analyst, I want a set of well-defined tools to query the dataset, so that the agent can answer precise questions about categories, intents, counts, and examples.

#### Acceptance Criteria

1. THE Agent SHALL expose a `list_categories` Tool that returns the distinct list of Category values present in the Dataset.
2. THE Agent SHALL expose a `filter_by_intent` Tool that accepts an intent name and returns all Dataset rows matching that intent.
3. THE Agent SHALL expose a `filter_by_category` Tool that accepts a category name and returns all Dataset rows matching that category.
4. THE Agent SHALL expose a `count_rows` Tool that accepts an optional category and optional intent filter and returns the integer count of matching rows.
5. THE Agent SHALL expose a `show_examples` Tool that accepts a category or intent name and an integer N between 1 and 50, and returns up to N representative utterance examples from the matching rows.
6. THE Agent SHALL expose a `get_intent_distribution` Tool that accepts a category name and returns the count of rows per intent within that category.
7. THE Agent SHALL expose a `summarize_category` Tool that accepts a category name and returns a natural-language summary of the utterances in that category using a Nebius_LLM call.
8. WHEN a Tool receives a category or intent name that does not exist in the Dataset, THE Tool SHALL return a structured error response indicating the value was not found, rather than raising an unhandled exception.
9. WHEN a Tool is called, THE Tool SHALL have a Pydantic input schema defining all parameters with types and descriptions.

---

### Requirement 4: Multi-Step Reasoning

**User Story:** As a data analyst, I want the agent to chain multiple tool calls to answer complex questions, so that I can get answers that require combining filtering, counting, and summarization.

#### Acceptance Criteria

1. WHEN a query requires combining results from multiple Tools (e.g., filter then count), THE Agent SHALL invoke the Tools sequentially within a single reasoning trace.
2. THE Agent SHALL enforce a maximum of 15 reasoning iterations per query to prevent infinite loops.
3. WHEN the Agent reaches the maximum iteration limit, THE Agent SHALL present the final answer if one has been produced, otherwise THE Agent SHALL emit a graceful fallback message explaining that the query could not be completed within the iteration limit.
4. THE Agent SHALL print each tool call name and its observation to the CLI output as each reasoning step completes, before the final answer is produced.
5. WHEN the Agent produces a final answer, THE Agent SHALL present it as a clear, human-readable response after the reasoning trace.

---

### Requirement 5: CLI Interface

**User Story:** As a user, I want to interact with the agent through a command-line interface, so that I can query the dataset interactively from my terminal.

#### Acceptance Criteria

1. THE CLI SHALL be launched with the command `python main.py` and SHALL enter an interactive query loop.
2. THE CLI SHALL accept an optional `--session` argument (e.g., `python main.py --session my_session`) to specify the session ID for conversation persistence.
3. WHEN no `--session` argument is provided, THE CLI SHALL generate and display a new session ID for the current session.
4. THE CLI SHALL print each reasoning step (tool call name and observation) to standard output before printing the final answer.
5. WHEN the user types `exit` or `quit`, THE CLI SHALL terminate the interactive loop gracefully.
6. THE CLI SHALL accept a `--user` argument to specify the user identifier for User_Profile loading and persistence.

---

### Requirement 6: Conversation Memory

**User Story:** As a user, I want my conversation history to persist across restarts, so that I can resume a session and the agent remembers what we discussed.

#### Acceptance Criteria

1. THE Agent SHALL use a SqliteSaver or PostgresSaver Checkpointer (not MemorySaver) to persist conversation state.
2. WHEN a session ID is provided via `--session`, THE Agent SHALL restore the full conversation history for that session from the Checkpointer.
3. WHEN a follow-up query references a prior turn (e.g., "How about that category?"), THE Agent SHALL resolve the reference using the restored conversation history.
4. WHEN any state-changing event occurs (new user message, agent response, session initialization, or configuration update), THE Checkpointer SHALL persist the updated state before the Agent returns its response.
5. IF Checkpointer persistence fails for any state-changing event, THEN THE Agent SHALL return an error to the user and SHALL NOT acknowledge the turn as completed.
6. THE Agent SHALL store checkpoints in a local SQLite file (default: `checkpoints.db`) whose path is configurable via an environment variable or CLI argument.

---

### Requirement 7: User Profile

**User Story:** As a user, I want the agent to remember my name, interests, and preferences across sessions, so that it can personalize responses and recall what it knows about me.

#### Acceptance Criteria

1. THE Agent SHALL maintain a User_Profile per user identifier that stores at minimum: name, list of frequent topics, and free-form preferences.
2. WHEN the user provides their name during a session, THE Agent SHALL update the User_Profile with the provided name.
3. WHEN the user asks "What do you remember about me?", THE Agent SHALL respond with the contents of the User_Profile for the current user.
4. WHEN the same intent or category is queried 3 or more times within a session, THE Agent SHALL add that intent or category to the frequent topics list in the User_Profile.
5. THE User_Profile SHALL persist across restarts in a per-user file or via the Checkpointer, keyed by the user identifier supplied via `--user`.
6. WHEN no `--user` argument is provided, THE Agent SHALL use a default user identifier and load the corresponding User_Profile.

---

### Requirement 8: MCP Server

**User Story:** As a developer, I want the agent's tools exposed as MCP-compatible endpoints, so that external clients can call them programmatically.

#### Acceptance Criteria

1. THE MCP_Server SHALL expose at least 3 of the core dataset Tools as MCP tools using the FastMCP framework.
2. WHEN an MCP client connects and calls an exposed tool, THE MCP_Server SHALL execute the tool against the loaded Dataset and return the result.
3. THE MCP_Server SHALL be startable independently of the CLI with a documented command (e.g., `python mcp_server.py`).
4. THE README SHALL include a section showing how to connect an MCP client and call at least one exposed tool with an example request and response.
5. WHEN an MCP tool call receives invalid parameters, THE MCP_Server SHALL return a structured error response with a descriptive message.

---

### Requirement 9: LLM Provider Constraint

**User Story:** As a system operator, I want all LLM calls to go exclusively through Nebius Token Factory, so that the system complies with the assignment's model usage policy.

#### Acceptance Criteria

1. THE Agent SHALL make all LLM calls exclusively through the Nebius Token Factory API.
2. THE Agent SHALL NOT call any other LLM provider API (e.g., OpenAI, Anthropic, HuggingFace Inference API) directly.
3. THE Agent SHALL read the Nebius API key from an environment variable (e.g., `NEBIUS_API_KEY`) and SHALL NOT hard-code credentials in source files.
4. IF the Agent cannot successfully read the Nebius API key from the environment for any reason (variable unset, empty value, or read failure), THEN THE Agent SHALL emit a descriptive error and exit with a non-zero status code.
5. THE README SHALL document which Nebius model(s) are used and justify the model choice.

---

### Requirement 10: Project Structure and Documentation

**User Story:** As a developer, I want a well-structured project with clear documentation, so that I can set up, run, and understand the system quickly.

#### Acceptance Criteria

1. THE project SHALL include a `requirements.txt` file listing all dependencies with pinned version numbers.
2. THE project SHALL include a `README.md` covering: setup instructions, CLI usage examples, MCP connection instructions, architecture overview, and model choice justification.
3. THE Agent source code SHALL use meaningful variable and function names, type hints on all public functions, and docstrings on all public functions and classes.
4. THE project repository name SHALL follow the pattern `From_AI_Model_to_AI_Agent_Assignment_3_<StudentNames>`.

---

### Requirement 11: Streamlit UI (Bonus)

**User Story:** As a non-technical user, I want a web-based chat interface, so that I can interact with the agent without using the command line.

#### Acceptance Criteria

1. WHERE the Streamlit UI is enabled, THE Streamlit_UI SHALL display a chat interface showing the full conversation history including agent reasoning steps.
2. WHERE the Streamlit UI is enabled, THE Streamlit_UI SHALL provide a session ID input field in the sidebar to allow the user to specify or restore a session.
3. WHERE the Streamlit UI is enabled, WHEN the user submits a query, THE Streamlit_UI SHALL display each reasoning step (tool call and observation) as it is produced before showing the final answer.
4. WHERE the Streamlit UI is enabled, THE Streamlit_UI SHALL be launchable with `streamlit run app.py` and SHALL connect to the same Agent and Checkpointer used by the CLI.

---

### Requirement 12: Query Recommender (Bonus)

**User Story:** As a user, I want the agent to suggest follow-up queries based on my history and profile, so that I can discover interesting analyses I might not have thought of.

#### Acceptance Criteria

1. WHERE the Query_Recommender is enabled, WHEN the user asks "What should I query next?", THE Query_Recommender SHALL generate at least 3 follow-up query suggestions based on the current session history and User_Profile.
2. WHERE the Query_Recommender is enabled, THE Agent SHALL present the suggestions and wait for the user to select, refine, or reject them before executing any query.
3. WHERE the Query_Recommender is enabled, WHEN the user confirms a suggested query, THE Agent SHALL execute it through the normal query routing path.
4. WHERE the Query_Recommender is enabled, WHEN the user refines a suggestion, THE Agent SHALL incorporate the refinement and confirm the updated query with the user before executing it.
5. WHERE the Query_Recommender is enabled, THE Query_Recommender SHALL NOT execute any query without explicit user confirmation.
