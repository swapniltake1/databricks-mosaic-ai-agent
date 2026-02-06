# Databricks notebook source
# MAGIC %md
# MAGIC # Refactored Quickstart: Build a Mosaic AI Agent on Databricks
# MAGIC This notebook demonstrates a cleaner, production-style agent structure:
# MAGIC - Workspace-aware LLM endpoint discovery
# MAGIC - Structured logging and MLflow tracing
# MAGIC - Tool-enabled multi-turn agent loop with Unity Catalog function tools
# MAGIC - Reusable class-based design for easier extension

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow databricks-openai databricks-agents databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import logging
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksFunctionClient, UCFunctionToolkit

warnings.filterwarnings("ignore")

# COMMAND ----------

# DBTITLE 1,Logger Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("mosaic-agent")
logger.info("Logger initialized")

# COMMAND ----------

# DBTITLE 1,Agent Configuration
@dataclass
class AgentConfig:
    candidate_endpoints: List[str]
    max_tool_iterations: int = 4


DEFAULT_CONFIG = AgentConfig(
    candidate_endpoints=[
        "databricks-claude-3-7-sonnet",
        "databricks-meta-llama-3-3-70b-instruct",
    ]
)

# COMMAND ----------

# DBTITLE 1,Mosaic Agent Implementation
class DatabricksMosaicAgent:
    """Tool-enabled Databricks Mosaic AI agent with MLflow instrumentation."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.workspace = WorkspaceClient()
        self.openai_client = self.workspace.serving_endpoints.get_open_ai_client()
        self.function_client = DatabricksFunctionClient()
        self.toolkit = UCFunctionToolkit(
            function_names=["system.ai.python_exec"],
            client=self.function_client,
        )
        self.tools = self.toolkit.tools
        self.model = self._resolve_model_endpoint()

    def _is_endpoint_available(self, endpoint_name: str) -> bool:
        """Smoke test endpoint by sending a short completion request."""
        try:
            self.openai_client.chat.completions.create(
                model=endpoint_name,
                messages=[{"role": "user", "content": "health-check"}],
            )
            return True
        except Exception:
            return False

    def _resolve_model_endpoint(self) -> str:
        """Select the first available model endpoint from candidates."""
        for endpoint in self.config.candidate_endpoints:
            if self._is_endpoint_available(endpoint):
                logger.info("Selected LLM endpoint: %s", endpoint)
                return endpoint

        raise ValueError(
            "No candidate LLM endpoint is available. "
            "Update AgentConfig.candidate_endpoints to a valid serving endpoint."
        )

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a Unity Catalog tool function and return serialized output."""
        if tool_name != "system__ai__python_exec":
            raise ValueError(f"Unknown tool requested by model: {tool_name}")

        tool_response = self.function_client.execute_function(
            "system.ai.python_exec", parameters=arguments
        )
        return tool_response.value

    def run_llm(self, prompt: str) -> str:
        """Run a direct LLM call (no tools), with logging and latency metrics."""
        start_time = time.time()
        logger.info("run_llm started | prompt=%s", prompt)

        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content

        latency = time.time() - start_time
        mlflow.log_param("llm_model", self.model)
        mlflow.log_metric("llm_latency_sec", latency)
        logger.info("run_llm completed in %.2fs", latency)

        return content

    def run_agent(self, prompt: str) -> List[Dict[str, Any]]:
        """Run a multi-turn tool-capable agent loop until final assistant response."""
        start_time = time.time()
        logger.info("run_agent started | prompt=%s", prompt)

        conversation: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
        emitted_messages: List[Dict[str, Any]] = []
        tool_calls = 0

        for _ in range(self.config.max_tool_iterations):
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=conversation,
                tools=self.tools,
            )

            message = response.choices[0].message
            assistant_message = message.to_dict()
            emitted_messages.append(assistant_message)
            conversation.append(assistant_message)

            # If no tool call exists, assistant produced final answer.
            if not message.tool_calls:
                break

            for tool_call in message.tool_calls:
                tool_calls += 1
                arguments = json.loads(tool_call.function.arguments)
                logger.info("Executing tool=%s args=%s", tool_call.function.name, arguments)

                tool_output = self._execute_tool(tool_call.function.name, arguments)
                tool_message = {
                    "role": "tool",
                    "name": tool_call.function.name,
                    "tool_call_id": tool_call.id,
                    "content": tool_output,
                }
                conversation.append(tool_message)
                emitted_messages.append(tool_message)

        latency = time.time() - start_time
        mlflow.log_param("agent_model", self.model)
        mlflow.log_metric("agent_latency_sec", latency)
        mlflow.log_metric("tool_calls", tool_calls)

        logger.info("run_agent completed in %.2fs | tool_calls=%s", latency, tool_calls)
        return emitted_messages

# COMMAND ----------

# DBTITLE 1,Initialize Agent + Enable Autolog
mlflow.openai.autolog()
agent = DatabricksMosaicAgent(config=DEFAULT_CONFIG)

# COMMAND ----------

# DBTITLE 1,Direct LLM Test
with mlflow.start_run(run_name="llm-test-run", nested=True):
    simple_answer = agent.run_llm("What is the capital of India?")
    print(simple_answer)

# COMMAND ----------

# DBTITLE 1,Tool-enabled Agent Test
with mlflow.start_run(run_name="agent-test-run", nested=True):
    agent_trace = agent.run_agent(
        "Use python to compute the square root of 92 and explain the result in one sentence."
    )

for message in agent_trace:
    role = message.get("role", "unknown")
    content = message.get("content", "")
    print(f"{role}: {content}")
