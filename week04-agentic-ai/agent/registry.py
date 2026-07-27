from typing import Any, Dict, List, Optional


class ToolRegistry:

    def __init__(self):
        self._tools: Dict[str, Any] = {}

    def register(self, tool_instance: Any) -> None:
        """Register a tool instance."""
        if not hasattr(tool_instance, "name") or not hasattr(
            tool_instance, "parameters"
        ):
            raise ValueError(
                "Registered tool must have 'name' and 'parameters' attributes."
            )
        self._tools[tool_instance.name] = tool_instance

    def get_tool(self, name: str) -> Optional[Any]:
        return self._tools.get(name)

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Returns tool definitions formatted as JSON Schemas for the LLM."""
        schemas = []
        for name, tool in self._tools.items():
            schemas.append(
                {
                    "name": tool.name,
                    "description": getattr(
                        tool, "description", "No description provided."
                    ),
                    "parameters": tool.parameters,
                }
            )
        return schemas

    def validate_call(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> List[str]:
        """Validates tool existence and required schema keys."""
        errors = []
        if tool_name not in self._tools:
            return [f"Tool '{tool_name}' is not registered in ToolRegistry."]

        tool = self._tools[tool_name]
        schema = getattr(tool, "parameters", {})
        required_keys = schema.get("required", [])

        for req in required_keys:
            if req not in arguments:
                errors.append(
                    f"Missing required argument '{req}' for tool '{tool_name}'."
                )

        return errors