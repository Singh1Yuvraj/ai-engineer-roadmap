"""
agent/memory.py
Execution history and conversation memory store for the Legal AI Agent.
"""

from typing import List, Dict, Any


class AgentMemory:
    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def add_user_message(self, message: str):
        """Logs incoming user queries."""
        self.history.append({"role": "user", "content": message})

    def add_observation(self, tool_name: str, tool_input: Any, output: Any):
        """Logs tool execution inputs and observations."""
        self.history.append({
            "role": "observation",
            "tool": tool_name,
            "input": tool_input,
            "output": output
        })

    def add_final_answer(self, answer: str):
        """Logs final response payload."""
        self.history.append({"role": "assistant", "content": answer})

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns the full execution trace."""
        return self.history

    def clear(self):
        """Clears memory state for new runs."""
        self.history = []