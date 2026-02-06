# Databricks Mosaic AI Agent Starter

A practical starter project for building a **tool-enabled AI agent** on Databricks using the **Mosaic AI Agent Framework**, Databricks Model Serving, Unity Catalog functions, and MLflow tracing.

## What this project includes

- A Databricks notebook-exported Python file that demonstrates:
  - selecting an available LLM serving endpoint,
  - making direct LLM calls,
  - running a multi-turn tool-enabled agent loop,
  - logging latency/tool metrics with MLflow.
- A class-based implementation (`DatabricksMosaicAgent`) intended to be easier to maintain and extend than ad-hoc notebook code.

Project file layout:

- `main/first_ai_agent_in_databricks.py` – main notebook-exported script with setup + agent implementation.

## Prerequisites

Before running this in Databricks, ensure:

1. You have access to a Databricks workspace with:
   - Model Serving enabled,
   - at least one compatible chat endpoint (for example, Claude or Llama endpoints),
   - Unity Catalog enabled (for `system.ai.python_exec` function access).
2. You can install Python dependencies in your notebook environment.
3. You have permissions to run MLflow experiments/runs.

## Dependencies

The notebook installs:

- `mlflow`
- `databricks-openai`
- `databricks-agents`
- `databricks-sdk`

These are installed in-notebook with `%pip` and followed by a Python restart.

## Quick start

1. Import the file into a Databricks notebook (or run it as a notebook-exported script in Databricks).
2. Run cells top-to-bottom.
3. The agent will:
   - initialize logging,
   - choose the first reachable LLM endpoint from configured candidates,
   - run a direct LLM test,
   - run a tool-enabled agent test that can execute Python via Unity Catalog tool calls.

## How the agent works

### 1) Configuration

`AgentConfig` controls:

- `candidate_endpoints`: ordered list of serving endpoint names to try.
- `max_tool_iterations`: upper bound for tool-loop rounds.

### 2) Endpoint resolution

`DatabricksMosaicAgent` performs a lightweight health-check completion against candidate endpoints and picks the first available one.

### 3) Tool execution

The agent registers the built-in Unity Catalog tool:

- `system.ai.python_exec`

When the model emits tool calls, the agent executes tool arguments and appends tool results back into the conversation.

### 4) Observability

The notebook enables:

- `mlflow.openai.autolog()` for LLM trace capture,
- custom metrics/params including model identity, latencies, and tool-call count.

## Configuration tips

- If your workspace uses different endpoint names, update `DEFAULT_CONFIG.candidate_endpoints`.
- Increase `max_tool_iterations` only when needed; lower values reduce latency/cost risk.
- For production, consider adding:
  - endpoint fallback strategies by region/tier,
  - stricter tool argument validation,
  - prompt/version tracking,
  - unit tests around tool routing logic.

## Troubleshooting

- **No endpoint found**: verify endpoint names and serving permissions.
- **Tool call fails**: ensure Unity Catalog is enabled and `system.ai.python_exec` is accessible to your principal.
- **No MLflow traces**: verify experiment permissions and runtime support for autologging.

## License

This project is provided under the terms in `LICENSE`.
