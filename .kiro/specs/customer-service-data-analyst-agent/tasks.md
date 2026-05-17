# Implementation Plan: Customer Service Data Analyst Agent

Convert the feature design into a series of prompts for a code-generation LLM that will implement each step with incremental progress. Make sure that each prompt builds on the previous prompts, and ends with wiring things together. There should be no hanging or orphaned code that isn't integrated into a previous step. Focus ONLY on tasks that involve writing, modifying, or testing code.

## Overview

The implementation follows the layered architecture from the design document:

1. **Foundations**: project scaffolding, configuration, Nebius LLM factory, dataset loader.
2. **Domain layer**: Pydantic tool schemas and the seven dataset tools.
3. **Persistence**: User Profile store and SqliteSaver checkpointer factory.
4. **Graph layer**: Query Router, decline/summarize/profile nodes, full LangGraph assembly with `create_react_agent` and the iteration cap.
5. **Frontends**: CLI entry point, FastMCP server, optional Streamlit UI, optional Query Recommender.
6. **Verification**: a property-based test suite covering all 16 correctness properties plus example/integration tests for non-PBT-amenable criteria.
7. **Documentation**: README sections covering setup, CLI, MCP, architecture, and model justification.

All code is Python 3.11+. All LLM calls go through a single Nebius factory (`src/csa_agent/llm.py`) by construction, satisfying Requirement 9 / Property 14.

## Tasks

- [x] 1. Set up project scaffolding and dependency manifest
  - [x] 1.1 Create directory tree, dependency manifest, gitignore, and README skeleton
    - Create the full directory tree from the design's "Project Layout": `src/csa_agent/`, `src/csa_agent/tools/`, `tests/`, `data/`, `profiles/`.
    - Create empty `__init__.py` files in `src/csa_agent/`, `src/csa_agent/tools/`, and `tests/`.
    - Create `requirements.txt` with pinned versions for: `langgraph`, `langchain`, `langchain-openai`, `langgraph-checkpoint-sqlite`, `pandas`, `pydantic>=2`, `python-dotenv`, `fastmcp`, `streamlit`, `pytest`, `pytest-cov`, `hypothesis`, `pytest-httpx`.
    - Add a `.gitignore` that excludes `checkpoints.db`, `profiles/`, `__pycache__/`, `.env`, `data/*.csv`.
    - Create a `README.md` skeleton with empty sections for Setup, CLI Usage, MCP Connection, Architecture, and Model Justification (filled in by task 17.1).
    - _Requirements: 10.1, 10.2_

- [x] 2. Implement configuration and Nebius LLM factory
  - [x] 2.1 Implement `src/csa_agent/config.py` settings loader
    - Define a `Settings` pydantic model exposing `nebius_api_key`, `nebius_base_url`, `nebius_model`, `dataset_path`, `checkpoint_db`, `profile_dir`, `max_iterations` with the defaults from the design's Configuration table.
    - Implement `get_settings()` that reads from environment (with `python-dotenv` for `.env`), trims whitespace, treats empty strings as missing, and exits with a non-zero status and a descriptive stderr message when `NEBIUS_API_KEY` is missing or empty.
    - Cache the `Settings` instance so subsequent calls are cheap.
    - _Requirements: 6.6, 9.3, 9.4_

  - [x] 2.2 Implement `src/csa_agent/llm.py` Nebius factory
    - Implement `get_llm(temperature=0.0, model=None) -> ChatOpenAI` that constructs a `langchain_openai.ChatOpenAI` with `base_url=settings.nebius_base_url`, `api_key=settings.nebius_api_key`, `model=model or settings.nebius_model`.
    - This factory is the only LLM constructor in the codebase. Add a module-level docstring documenting that constraint (referenced by Property 14).
    - _Requirements: 9.1, 9.2_

  - [ ]* 2.3 Write unit tests for `Settings` startup behavior
    - Cover: missing/empty `NEBIUS_API_KEY` triggers `SystemExit(1)`; valid env yields a `Settings` instance; defaults match the design table.
    - _Requirements: 9.3, 9.4, 6.6_

- [x] 3. Implement dataset loader and validation
  - [x] 3.1 Implement `src/csa_agent/dataset.py`
    - Define `REQUIRED_COLUMNS = {"utterance", "category", "intent"}` and `load_dataset(path: str) -> pandas.DataFrame`.
    - Support both CSV and Parquet by file extension.
    - Raise `FileNotFoundError` with a descriptive message when the path is missing; raise `ValueError` listing missing columns when validation fails.
    - Provide a `get_dataset()` accessor that loads the dataset once and caches it as a module-level singleton so tools share a single in-memory frame.
    - _Requirements: 1.1, 1.3, 1.5_

  - [ ]* 3.2 Write example tests for dataset loader
    - Use small fixture CSVs to cover: happy-path load, missing file → `FileNotFoundError`, missing-column file → `ValueError` listing the missing columns, repeated `get_dataset()` calls return the same object reference.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 4. Implement Pydantic tool input schemas
  - [x] 4.1 Create `src/csa_agent/tools/schemas.py`
    - Define `ListCategoriesInput`, `FilterByIntentInput`, `FilterByCategoryInput`, `CountRowsInput`, `ShowExamplesInput`, `GetIntentDistributionInput`, `SummarizeCategoryInput` exactly as specified in the design "Pydantic Tool Input Schemas" subsection.
    - Apply `Field(..., ge=1, le=50)` to `ShowExamplesInput.n` so Pydantic rejects out-of-range values at validation time.
    - Define a `ToolError` Pydantic model with `error: str`, `message: str`, `value: str | None`.
    - _Requirements: 3.9, 3.5_

- [x] 5. Implement the seven dataset tools
  - [x] 5.1 Implement `src/csa_agent/tools/core.py` with a `build_tools(df)` factory
    - Implement `list_categories`, `filter_by_intent`, `filter_by_category`, `count_rows`, `show_examples`, `get_intent_distribution` as pure functions over the captured `df`.
    - `filter_by_*` results are capped at 100 rows (per design); `count_rows` always returns the full count.
    - `show_examples` clamps `n` to `[1, 50]` defensively even though the schema validates it.
    - On unknown category/intent, every tool returns a `ToolError`-shaped dict (`{"error": ..., "message": ..., "value": ...}`) and never raises.
    - Wrap each function with LangChain's `@tool` decorator using the matching Pydantic input schema, and return them as `list[BaseTool]` from `build_tools(df)`.
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.8, 3.9_

  - [x] 5.2 Implement the `summarize_category` tool
    - In the same `build_tools(df)` factory, add `summarize_category` which: validates the category exists (returns `ToolError` if not), samples representative utterances from the category, and produces a natural-language summary via `get_llm()` with a grounded prompt that includes the sampled utterances verbatim.
    - _Requirements: 3.7, 3.8, 9.1_

  - [ ]* 5.3 Write property test for `list_categories` (Property 1)
    - **Property 1: list_categories is the deduplicated category column**
    - **Validates: Requirements 3.1**
    - Use a Hypothesis `dataframes_st()` strategy in `tests/strategies.py` and assert `set(list_categories(df)) == set(df["category"])` and no duplicates.

  - [ ]* 5.4 Write property test for `filter_by_intent` and `filter_by_category` (Property 2)
    - **Property 2: filter_by_* returns exactly the matching rows**
    - **Validates: Requirements 3.2, 3.3**
    - Compare each tool's output against the equivalent `df[df.col == value]`.

  - [ ]* 5.5 Write property test for `count_rows` (Property 3)
    - **Property 3: count_rows is consistent with the filter tools**
    - **Validates: Requirements 3.4**
    - Assert `count_rows(df, category, intent)` equals the size of the intersection of the filter outputs.

  - [ ]* 5.6 Write property test for `get_intent_distribution` (Property 4)
    - **Property 4: get_intent_distribution is consistent with count_rows**
    - **Validates: Requirements 3.6**
    - Assert sum of distribution values equals `count_rows(df, category=category)` and every key is an intent within the category.

  - [ ]* 5.7 Write property test for `show_examples` (Property 5)
    - **Property 5: show_examples is bounded and grounded**
    - **Validates: Requirements 3.5**
    - For random `n ∈ [1, 50]` and random filters, assert `len(result) <= min(n, matching_count)` and every returned utterance is in the matching subset.

  - [ ]* 5.8 Write property test for tool error handling (Property 6)
    - **Property 6: Tools return structured errors for missing values**
    - **Validates: Requirements 3.8**
    - Use a `non_existent_string_st(df)` strategy that prefixes a guaranteed-absent sentinel and assert every category/intent-taking tool returns a `ToolError`-shaped dict and does not raise.

- [x] 6. Checkpoint - foundations and tools complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement User Profile store
  - [x] 7.1 Implement `src/csa_agent/profile.py`
    - Define `UserProfile` Pydantic model with `user_id`, `name`, `frequent_topics`, `preferences`, `topic_counts`, `created_at`, `updated_at` per the design.
    - Implement `load_profile(user_id)` that reads `{profile_dir}/{user_id}.json` or returns a fresh profile if the file is absent.
    - Implement `save_profile(profile)` using atomic write (temp file + `os.replace`) to avoid partial writes.
    - Implement `record_topic(profile, topic)` that increments `topic_counts[topic]` and adds `topic` to `frequent_topics` when the count reaches 3.
    - _Requirements: 7.1, 7.2, 7.4, 7.5, 7.6_

  - [ ]* 7.2 Write property test for profile round-trip and topic counter (Property 12)
    - **Property 12: Profile round-trip and topic counter**
    - **Validates: Requirements 7.4, 7.5**
    - Use a `user_profiles_st()` strategy: assert `load_profile(save_profile(p).user_id) == p` and that `t in frequent_topics` iff `topic_counts[t] >= 3` after any sequence of `record_topic` calls.

- [x] 8. Implement checkpointer factory
  - [x] 8.1 Implement `src/csa_agent/checkpointer.py`
    - Implement `get_checkpointer(db_path)` that returns a `SqliteSaver.from_conn_string(db_path)` context manager wrapper, or a `PostgresSaver` when `POSTGRES_URL` env var is set.
    - Validate that the parent directory of `db_path` exists and is writable; raise `OSError` with a descriptive message otherwise.
    - _Requirements: 6.1, 6.6_

  - [ ]* 8.2 Write property test for checkpointer message-order preservation (Property 13)
    - **Property 13: Checkpointer preserves message order across reopens**
    - **Validates: Requirements 6.2, 6.4**
    - Generate a `message_sequences_st()` list, write it to a thread, close and reopen the SqliteSaver pointing at the same file, and assert recovered messages equal the input in order.

- [x] 9. Implement Query Router node
  - [x] 9.1 Implement `src/csa_agent/router.py`
    - Define the `RouteLabel(str, Enum)` with `STRUCTURED`, `UNSTRUCTURED`, `OUT_OF_SCOPE` values.
    - Implement `classify_query(user_query, llm)` that uses `llm.with_structured_output(RouteLabel)` and a focused classification prompt with no tool bindings.
    - Coerce/clamp any unexpected raw output to a valid `RouteLabel` (defaulting to `OUT_OF_SCOPE` on parse failure) so the function is total.
    - _Requirements: 2.1, 2.5_

  - [ ]* 9.2 Write property test for router output well-formedness (Property 7)
    - **Property 7: Query Router output is well-formed**
    - **Validates: Requirements 2.1**
    - Patch the LLM with a `FakeChatModel` that returns arbitrary Hypothesis-generated strings; assert `classify_query` always returns a `RouteLabel` member.

- [x] 10. Implement decline, summarize, and profile graph nodes
  - [x] 10.1 Implement `src/csa_agent/nodes.py`
    - Implement `decline_node(state)` that appends the canonical refusal AIMessage and makes zero LLM/tool calls.
    - Implement `summarize_node(state)` as a small ReAct subgraph bound to `count_rows`, `show_examples`, `get_intent_distribution` plus a system prompt requiring grounded summaries.
    - Implement `load_user_profile_node(state, config)` that reads `config["configurable"]["user_id"]` and injects the loaded `UserProfile` into state.
    - Implement `update_profile_node(state, config)` that calls `record_topic` for any category/intent referenced in the turn, then `save_profile`. Failures here are logged but do not block the response.
    - _Requirements: 2.2, 2.4, 7.1, 7.3, 7.4_

  - [ ]* 10.2 Write integration test for decline path purity (Property 8)
    - **Property 8: Decline path is pure**
    - **Validates: Requirements 2.2, 2.5**
    - Patch `classify_query` to return `OUT_OF_SCOPE`; spy on the LLM factory and tool registry; assert zero LLM calls and zero tool invocations after the router.

- [x] 11. Assemble the LangGraph
  - [x] 11.1 Implement `src/csa_agent/graph.py` with `build_graph(...)`
    - Wire `__start__ → load_user_profile → query_router` with conditional edges to `decline_node`, `react_agent`, or `summarize_node` per the design's node graph.
    - Build the ReAct branch using `langgraph.prebuilt.create_react_agent(model=get_llm(), tools=build_tools(df), state_modifier=...)` with a `state_modifier` that injects the user profile context and a "ground answers in tool observations" instruction.
    - Apply a recursion limit of `settings.max_iterations` (15) on the ReAct subgraph and a graceful fallback message when the cap is reached.
    - All branches converge into `update_profile_node` then `__end__`.
    - Compile with the checkpointer from `get_checkpointer(...)` so every super-step persists.
    - Return the compiled graph plus a small streaming helper that yields `(tool_name, args, observation)` for each tool event and the final answer last.
    - _Requirements: 2.3, 2.4, 4.1, 4.2, 4.3, 4.4, 6.1, 6.4, 6.5_

  - [ ]* 11.2 Write integration test for routing branches (Property 9)
    - **Property 9: Routing matches classification**
    - **Validates: Requirements 2.3, 2.4**
    - Patch `classify_query` to return each `RouteLabel`; assert the next visited node matches (`structured` → ReAct, `unstructured` → summarize, `out_of_scope` → decline).

  - [ ]* 11.3 Write property test for ReAct iteration cap (Property 10)
    - **Property 10: ReAct agent terminates within the iteration cap**
    - **Validates: Requirements 4.2, 4.3**
    - Use a `FakeChatModel` that always emits a tool call; assert termination within 15 iterations and a non-empty final user-visible message.

  - [ ]* 11.4 Write property test for streaming order (Property 11)
    - **Property 11: Tool calls are streamed before the final answer**
    - **Validates: Requirements 4.4, 5.4, 11.3**
    - For runs with at least one tool call, assert every tool-call event index is strictly less than the final-answer event index.

  - [ ]* 11.5 Write property test for the LLM provider invariant (Property 14)
    - **Property 14: All LLM clients point at Nebius**
    - **Validates: Requirements 9.1, 9.2**
    - Combine: (a) `pytest-httpx` spy asserting outbound `base_url` matches `NEBIUS_BASE_URL` for every LLM call across a graph run; (b) static AST scan of `src/csa_agent/` rejecting any `ChatOpenAI(...)`, `ChatAnthropic`, `ChatGoogleGenerativeAI`, etc. constructor outside `llm.py`.

- [x] 12. Implement CLI entry point
  - [x] 12.1 Implement `main.py`
    - Parse `--session`, `--user`, `--checkpoint-db` with `argparse`. Default `--user` to `"default"`. When `--session` is omitted, generate a `uuid4()` and print it.
    - Build the graph once via `build_graph(...)` and enter an interactive `input()` loop.
    - For each query, call `graph.stream(..., config={"configurable": {"thread_id": session_id, "user_id": user_id}}, stream_mode="updates")` and print each tool call as `🔧 tool_name(args) → observation` before the final answer.
    - On `exit` or `quit`, break the loop cleanly. On `KeyboardInterrupt`, print a friendly goodbye and exit 0.
    - On checkpointer save failure mid-turn, surface an error to the user and do not acknowledge the turn as completed.
    - _Requirements: 4.4, 4.5, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 6.2, 6.5, 7.6_

  - [ ]* 12.2 Write CLI smoke test
    - Use `subprocess.Popen` with scripted stdin to verify: `python main.py` enters the loop, accepts `--session` and `--user`, prints a generated session ID when `--session` is omitted, and exits cleanly on `exit`/`quit`.
    - _Requirements: 5.1, 5.2, 5.3, 5.5, 5.6_

  - [ ]* 12.3 Write integration test for multi-step ReAct reasoning
    - With a scripted `FakeChatModel` that emits `filter_by_category` then `count_rows` before answering, assert both tool calls appear in the streamed updates in order.
    - _Requirements: 4.1_

  - [ ]* 12.4 Write integration test for "what do you remember about me?" flow
    - Pre-populate a profile, run a session asking "what do you remember about me?", assert the response includes the stored name/topics/preferences.
    - _Requirements: 7.3_

- [x] 13. Implement FastMCP server
  - [x] 13.1 Implement `mcp_server.py`
    - Initialize a FastMCP server, load the dataset via `get_dataset()`, and register at least 5 tools (`list_categories`, `count_rows`, `show_examples`, `filter_by_category`, `get_intent_distribution`) using `@mcp.tool()` with the same Pydantic input schemas from `tools/schemas.py`.
    - Make the file runnable with `python mcp_server.py` and configure host/port via env vars with sensible defaults.
    - Rely on FastMCP's Pydantic-driven validation so invalid inputs return a structured error.
    - _Requirements: 8.1, 8.2, 8.3, 8.5_

  - [ ]* 13.2 Write property test for MCP/direct tool equivalence (Property 15)
    - **Property 15: MCP tool calls match direct tool calls**
    - **Validates: Requirements 8.2, 8.5**
    - Spawn a FastMCP test client; for random valid inputs assert MCP result equals the direct tool result; for inputs violating the Pydantic schema assert a structured error response with no exception.

- [x] 14. Checkpoint - core agent, CLI, and MCP server complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 15. Implement Streamlit UI (bonus)
  - [x] 15.1 Implement `app.py`
    - Build a `st.chat_input` + `st.chat_message` layout. Sidebar contains a session ID text input and a user ID input.
    - On submit, invoke the same compiled graph used by the CLI; render streaming updates inside an `st.status` block so reasoning steps appear before the final answer.
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

  - [ ]* 15.2 Write Streamlit AppTest harness test
    - Use `streamlit.testing.v1.AppTest` to assert: the session sidebar input is rendered, submitting a query produces a chat message turn, and reasoning-step entries appear before the final answer.
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [x] 16. Implement Query Recommender (bonus)
  - [x] 16.1 Implement `src/csa_agent/recommender.py`
    - Detect the trigger phrase "What should I query next?" at the router level as a special intent.
    - Use `get_llm()` plus the loaded profile and recent message history to produce ≥ 3 follow-up suggestions.
    - Surface suggestions to the user and require explicit confirmation/refinement before invoking the normal routing path.
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

  - [ ]* 16.2 Write property test for recommender confirmation invariant (Property 16)
    - **Property 16: Recommender requires confirmation**
    - **Validates: Requirements 12.1, 12.2, 12.5**
    - For arbitrary suggestion sets, assert at least 3 suggestions are produced and zero downstream queries execute until an explicit confirmation event is observed.

- [x] 17. Write documentation and repository hygiene
  - [x] 17.1 Fill in README sections
    - Setup (env vars, `requirements.txt` install, dataset path).
    - CLI usage examples (`python main.py`, `--session`, `--user`, `--checkpoint-db`, `exit`/`quit`).
    - MCP connection instructions with a concrete example MCP client request and response for at least one exposed tool.
    - Architecture overview referencing the LangGraph node graph.
    - Model choice justification for `meta-llama/Meta-Llama-3.1-70B-Instruct` (or the chosen Nebius model).
    - _Requirements: 8.4, 9.5, 10.2_

  - [ ]* 17.2 Write repository hygiene tests
    - Assert `requirements.txt` has pinned versions for every entry.
    - Assert `README.md` contains all five required section headings (Setup, CLI Usage, MCP Connection, Architecture, Model Justification).
    - _Requirements: 10.1, 10.2_

- [x] 18. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP delivery.
- Each task references specific requirements for traceability.
- Property tests cover all 16 correctness properties from the design; each is its own sub-task placed close to the implementation it validates.
- Hypothesis strategies (`dataframes_st`, `non_existent_string_st`, `user_profiles_st`, `message_sequences_st`) live in `tests/strategies.py` and are introduced lazily by the first property test that needs them.
- A `FakeChatModel` test double replaces the real Nebius LLM in property tests to keep iterations cheap.
- Checkpoints exist at task 6 (foundations + tools), task 14 (core agent + CLI + MCP), and task 18 (final).
- Bonus tasks (Streamlit UI, Query Recommender) are isolated so they can be skipped without affecting the core deliverable.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1", "2.3", "3.1", "3.2", "4.1"] },
    { "id": 1, "tasks": ["2.2", "5.1", "7.1", "8.1"] },
    { "id": 2, "tasks": ["5.2", "5.3", "5.4", "5.5", "5.6", "5.7", "5.8", "7.2", "8.2", "9.1"] },
    { "id": 3, "tasks": ["9.2", "10.1", "13.1"] },
    { "id": 4, "tasks": ["10.2", "11.1", "13.2", "16.1"] },
    { "id": 5, "tasks": ["11.2", "11.3", "11.4", "11.5", "12.1", "15.1", "16.2"] },
    { "id": 6, "tasks": ["12.2", "12.3", "12.4", "15.2", "17.1"] },
    { "id": 7, "tasks": ["17.2"] }
  ]
}
```
