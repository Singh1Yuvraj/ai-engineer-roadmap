import json
import re
from typing import Any, Dict, List
# pyrefly: ignore [missing-import]
from agent.registry import ToolRegistry


class FunctionPlanner:

    def __init__(self, llm_client, registry: ToolRegistry):
        self.llm = llm_client
        self.registry = registry

    def create_plan(self, user_query: str) -> List[Dict[str, Any]]:
        schemas = self.registry.get_schemas()

        prompt = f"""You are an AI Function Calling Agent.
Available Tool Schemas:
{json.dumps(schemas, indent=2)}

State Variable Rule:
If a step depends on the output of a previous step, set that argument to "$last" or "$<tool_name>.output".

User Query: {user_query}

Return a valid JSON array of tool execution objects matching this structure:
[
  {{
    "tool": "<tool_name>",
    "arguments": {{ ... }}
  }}
]

Return ONLY valid JSON."""

        raw_response = self.llm.invoke(prompt)
        return self._parse_tool_calls(raw_response)

    def _parse_tool_calls(self, raw_response: str) -> List[Dict[str, Any]]:
        cleaned = raw_response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()

        try:
            plan = json.loads(cleaned)
            return plan if isinstance(plan, list) else [plan]
        except json.JSONDecodeError:
            return [{"tool": "search", "arguments": {"query": cleaned[:100]}}]