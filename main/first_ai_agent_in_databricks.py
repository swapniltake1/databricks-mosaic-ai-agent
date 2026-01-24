# Databricks notebook source
# MAGIC %md
# MAGIC # Quickstart: Build and test simple Agent using Mosaic AI Agent Framework
# MAGIC - This quickstart notebook demonstrates how to build and test a generative AI agent using Mosaic AI Agent Framework on Databricks
# MAGIC - Automatically log traces from LLM calls for ease of debugging using Mlflow
# MAGIC - The agent uses an LLM served on Databricks Foundation Model API 
# MAGIC - The agent has access to a single tool, the built-in Python code interpreter tool on Databricks Unity Catalog. It can use this tool to run LLM-generated code in order to respond to user questions. 
# MAGIC
# MAGIC We will use `databricks_openai` SDK to query the LLM endpoint.
# MAGIC

# COMMAND ----------

# MAGIC %pip list #check what is already installed

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow databricks-openai databricks-agents

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip show databricks-agents

# COMMAND ----------

# DBTITLE 1,Suppress warnings globally
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC lets add the loggin code hre 

# COMMAND ----------

import logging
import mlflow
import time

# -------------------------------
# Python Logger Configuration
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("mosaic-agent")

logger.info("Logging initialized successfully")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Pick the first LLM API available in your Databricks workspace

# COMMAND ----------

# The snippet below tries to pick the first LLM API available in your Databricks workspace
# from a set of candidates. You can override and simplify it
# to just specify LLM_ENDPOINT_NAME.
LLM_ENDPOINT_NAME = None

from databricks.sdk import WorkspaceClient
def is_endpoint_available(endpoint_name):
  try:
    client = WorkspaceClient().serving_endpoints.get_open_ai_client()
    client.chat.completions.create(model=endpoint_name, messages=[{"role": "user", "content": "What is AI?"}])
    return True
  except Exception:
    return False
  
client = WorkspaceClient()
for candidate_endpoint_name in ["databricks-claude-3-7-sonnet", "databricks-meta-llama-3-3-70b-instruct"]:
    if is_endpoint_available(candidate_endpoint_name):
      LLM_ENDPOINT_NAME = candidate_endpoint_name
assert LLM_ENDPOINT_NAME is not None, "Please specify LLM_ENDPOINT_NAME"

# COMMAND ----------

LLM_ENDPOINT_NAME

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lets first do a LLM call

# COMMAND ----------

import json
import mlflow
from databricks.sdk import WorkspaceClient

# Automatically log traces from LLM calls for ease of debugging
mlflow.openai.autolog()

# Get an OpenAI client configured to talk to Databricks model serving endpoints
# We'll use this to query an LLM in our agent
openai_client = WorkspaceClient().serving_endpoints.get_open_ai_client()

prompt = "What is the capital of india ?"

openai_client.chat.completions.create(
        model=LLM_ENDPOINT_NAME,
        messages=[{"role": "user", "content": prompt}],
    )

# COMMAND ----------

response = openai_client.chat.completions.create(
        model=LLM_ENDPOINT_NAME,
        messages=[{"role": "user", "content": prompt}],
    )

# COMMAND ----------

response

# COMMAND ----------

response.id

# COMMAND ----------

response.choices[0].message.content

# COMMAND ----------

# MAGIC %md
# MAGIC ### Now lets wrap in function

# COMMAND ----------

def run_llm(prompt):
    """
    Send a user prompt to the LLM and log inputs/outputs
    """
    start_time = time.time()
    logger.info(f"LLM call started | Prompt: {prompt}")

    try:
        response = openai_client.chat.completions.create(
            model=LLM_ENDPOINT_NAME,
            messages=[{"role": "user", "content": prompt}],
        )

        msg = response.choices[0].message
        latency = time.time() - start_time

        # MLflow logging
        mlflow.log_param("llm_model", LLM_ENDPOINT_NAME)
        mlflow.log_metric("llm_latency_sec", latency)

        logger.info(f"LLM response received in {latency:.2f}s")
        logger.debug(f"LLM raw response: {msg.content}")

        return [msg.to_dict()]

    except Exception as e:
        logger.error("LLM call failed", exc_info=True)
        mlflow.log_param("llm_error", str(e))
        raise


# COMMAND ----------

resp = run_llm("What is the capital of india ?")
for message in resp:
    print(f'{message["role"]}: {message["content"]}')


# COMMAND ----------

# MAGIC %md
# MAGIC ## Let's add tool and create a Agent

# COMMAND ----------

from databricks_openai import UCFunctionToolkit, DatabricksFunctionClient
from databricks.sdk import WorkspaceClient

# Automatically log traces from LLM calls for ease of debugging
mlflow.openai.autolog()

# Get an OpenAI client configured to talk to Databricks model serving endpoints
# We'll use this to query an LLM in our agent
openai_client = WorkspaceClient().serving_endpoints.get_open_ai_client()

# Load Databricks built-in tools (a stateless Python code interpreter tool)
client = DatabricksFunctionClient()
builtin_tools = UCFunctionToolkit(
    function_names=["system.ai.python_exec"], client=client
).tools
#for tool in builtin_tools:
#   del tool["function"]["strict"]


def call_tool(tool_name, parameters):
    if tool_name == "system__ai__python_exec":
        return DatabricksFunctionClient().execute_function(
            "system.ai.python_exec", parameters=parameters
        )
    raise ValueError(f"Unknown tool: {tool_name}")


def run_agent(prompt):
    """
    Agent with tool usage + full logging
    """
    logger.info(f"Agent invoked | User prompt: {prompt}")
    start_time = time.time()

    result_msgs = []

    try:
        response = openai_client.chat.completions.create(
            model=LLM_ENDPOINT_NAME,
            messages=[{"role": "user", "content": prompt}],
            tools=builtin_tools,
        )

        msg = response.choices[0].message
        result_msgs.append(msg.to_dict())

        logger.info("LLM responded")
        logger.info(f"Assistant message: {msg.content}")

        # Tool execution logging
        if msg.tool_calls:
            call = msg.tool_calls[0]
            logger.info(f"Tool called: {call.function.name}")
            logger.debug(f"Tool arguments: {call.function.arguments}")

            tool_result = call_tool(
                call.function.name,
                json.loads(call.function.arguments)
            )

            logger.info("Tool execution completed")

            result_msgs.append({
                "role": "tool",
                "content": tool_result.value,
                "name": call.function.name,
                "tool_call_id": call.id,
            })

            mlflow.log_metric("tool_calls", 1)
        else:
            mlflow.log_metric("tool_calls", 0)

        latency = time.time() - start_time
        mlflow.log_metric("agent_latency_sec", latency)

        logger.info(f"Agent completed in {latency:.2f}s")

        return result_msgs

    except Exception as e:
        logger.error("Agent execution failed", exc_info=True)
        mlflow.log_param("agent_error", str(e))
        raise


# COMMAND ----------

# DBTITLE 1,Cell 23
with mlflow.start_run(run_name="agent-test-run", nested=True):
    answer = run_agent("What is the square root of 9?")
    for message in answer:
        print(f'{message["role"]}: {message["content"]}')

# COMMAND ----------

import json
answer = run_agent("What is the square root of 92?")
for message in answer:
    print(f'{message["role"]}: {message["content"]}')
