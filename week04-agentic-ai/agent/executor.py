"""
agent/executor.py
Execution engine that runs plans through registered tools.
"""

from typing import List, Dict, Any
# pyrefly: ignore [missing-import]
from agent.registry import ToolRegistry
# pyrefly: ignore [missing-import]
from agent.memory import AgentMemory


class PlanExecutor:
    def __init__(self, registry: ToolRegistry, memory: AgentMemory):
        self.registry = registry
        self.memory = memory

    def execute_plan(self, plan: List[Dict[str, Any]]) -> Any:
        """Executes a series of planned steps sequentially."""
        last_output = None

        for step in plan:
            tool_name = step["tool"]
            tool_input = step["input"]

            # If input specifies using prior results, chain the last output
            if tool_input == "use_search_results" and last_output is not None:
                tool_input = last_output

            tool = self.registry.get_tool(tool_name)
            output = tool.run(tool_input)

            # Record step observation into memory
            self.memory.add_observation(
                tool_name=tool_name,
                tool_input=tool_input,
                output=output
            )
            last_output = output

        return last_output