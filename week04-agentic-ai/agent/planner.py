import json
import re
from typing import Any, Dict, List


class LLMPlanner:

    def __init__(self, llm_client, prompt_path: str = "prompts/planner_prompt.txt"):
        self.llm = llm_client
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    def create_plan(self, user_query: str) -> List[Dict[str, Any]]:
        full_prompt = f"{self.system_prompt}\n\nUser Query: {user_query}\nPlan:"

        # LLM Call strictly for planning
        raw_response = self.llm.invoke(full_prompt)

        return self._parse_json_plan(raw_response)

    def _parse_json_plan(self, raw_response: str) -> List[Dict[str, Any]]:
        cleaned = raw_response.strip()

        # Strip markdown fences if the LLM ignores instructions
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()

        try:
            plan = json.loads(cleaned)
            if isinstance(plan, list):
                return plan
            elif isinstance(plan, dict):
                return [plan]
            else:
                raise ValueError("Parsed JSON is not a list or dict.")
        except json.JSONDecodeError as e:
            print(f"⚠️ [Planner Error] Could not parse JSON plan: {e}")
            # Fallback tool call if plan generation failed
            return [
                {
                    "tool": "search",
                    "input": {"query": cleaned[:100]},
                }
            ]