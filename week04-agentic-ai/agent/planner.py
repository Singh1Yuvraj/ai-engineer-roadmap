import json
import re
from typing import Any, Dict
# pyrefly: ignore [missing-import]
from agent.registry import ToolRegistry
# pyrefly: ignore [missing-import]
from agent.state import AgentState


class ReasoningPlanner:

    def __init__(self, llm_client, registry: ToolRegistry):
        self.llm = llm_client
        self.registry = registry

    def plan_next_step(self, state: AgentState) -> Dict[str, Any]:
        schemas = self.registry.get_schemas()

        prompt = f"""You are an Advanced AI Reasoning Agent for Legal Intelligence.

Conversation History:
{state.conversation_context}

User Query: {state.user_query}

Working Memory / Scratchpad:
{state.get_scratchpad_formatted()}

Available Tool Schemas:
{json.dumps(schemas, indent=2)}

INSTRUCTIONS:
1. Analyze the query and working memory.
2. REFLECTION: Do you have sufficient information from previous steps to answer the user fully?
3. If YES: return "action": "FINAL_ANSWER" with a "thought".
4. If NO: return "action": "TOOL_CALL" specifying the "tool" and "arguments" required.
   - Use "$last" as an argument value if passing the output of the most recent step.

OUTPUT FORMAT (JSON only, no markdown blocks):
{{
  "thought": "<Reasoning step or reflection on current progress>",
  "action": "TOOL_CALL" | "FINAL_ANSWER",
  "tool": "<tool_name_if_action_is_TOOL_CALL>",
  "arguments": {{ ... }}
}}"""

        raw_response = self.llm.invoke(prompt)
        return self._parse_response(raw_response)

    def _parse_response(self, raw_response: str) -> Dict[str, Any]:
        cleaned = raw_response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "thought": "Failed to parse JSON; falling back to search.",
                "action": "TOOL_CALL",
                "tool": "search",
                "arguments": {"query": cleaned[:100]},
            }