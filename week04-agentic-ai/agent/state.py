from typing import Any, Dict, List, Optional


class AgentState:

    def __init__(
        self, user_query: str, conversation_context: str = "No prior history."
    ):
        self.user_query: str = user_query
        self.conversation_context: str = conversation_context
        self.scratchpad: List[str] = []
        self.tool_results: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.last_result: Optional[Any] = None
        self.is_complete: bool = False
        self.final_answer: str = ""
        self.step_count: int = 0
        self.max_steps: int = 5

    def add_scratchpad_note(self, thought: str) -> None:
        self.scratchpad.append(f"Thought: {thought}")

    def record_step(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        output: Any,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        step_record = {
            "step": self.step_count,
            "tool": tool_name,
            "arguments": arguments,
            "output": output,
            "status": status,
            "error": error,
        }
        self.history.append(step_record)
        self.tool_results[tool_name] = output
        self.last_result = output
        self.scratchpad.append(f"Observation [{tool_name}]: {output}")

    def resolve_arguments(
        self, raw_arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolves state variables like '$last' or '$tool_name.output' into real runtime values."""
        resolved = {}
        for key, val in raw_arguments.items():
            if isinstance(val, str) and val.startswith("$"):
                if val == "$last":
                    resolved[key] = (
                        self.last_result if self.last_result is not None else ""
                    )
                elif val.startswith("$") and ".output" in val:
                    target_tool = val[1:].split(".output")[0]
                    resolved[key] = self.tool_results.get(
                        target_tool, self.last_result
                    )
                else:
                    resolved[key] = val
            else:
                resolved[key] = val
        return resolved

    def get_scratchpad_formatted(self) -> str:
        if not self.scratchpad:
            return "No prior thoughts or observations."
        return "\n".join(self.scratchpad)